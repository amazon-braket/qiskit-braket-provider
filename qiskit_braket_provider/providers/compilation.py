"""Compilation pipeline for Qiskit circuits targeting Braket devices.

This module contains the internal compilation pipeline used by to_braket()
and the run() endpoints, including verbatim box handling and transpilation.
"""

import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit import Barrier, BoxOp, Measure
from qiskit.transpiler import PassManager, Target

from braket.aws import AwsDevice
from braket.devices import Device
from qiskit_braket_provider.providers.gate_mappings import (
    _BRAKET_GATE_NAME_TO_QISKIT_GATE,
    _BRAKET_TO_QISKIT_NAMES,
    _BRAKET_VERBATIM_BOX_NAME,
)
from qiskit_braket_provider.providers.passes import (
    ExtractVerbatimBoxes,
    RestoreVerbatimBoxes,
)
from qiskit_braket_provider.providers.target import (
    _SubstitutedTarget,
    aws_device_to_target,
    local_simulator_to_target,
)

_T = TypeVar("_T")


@dataclass(frozen=True)
class _CompilationContext:
    """Internal result from _compile containing compiled circuits and resolved state."""

    circuits: list[QuantumCircuit]
    target: Target | None
    qubit_labels: Sequence[int] | None
    verbatim: bool | None
    basis_gates: Collection[str] | None
    angle_restrictions: Mapping[str, Mapping[int, set[float] | tuple[float, float]]] | None
    pass_manager: PassManager | None


def _default_target(circuits: Iterable[QuantumCircuit]) -> Target:
    num_qubits = max(circuit.num_qubits for circuit in circuits)
    target = Target(num_qubits=num_qubits)
    for braket_name, instruction in _BRAKET_GATE_NAME_TO_QISKIT_GATE.items():
        if name := _BRAKET_TO_QISKIT_NAMES.get(braket_name.lower()):
            target.add_instruction(instruction, name=name)
    target.add_instruction(Measure())
    target.add_instruction(Barrier(1))
    return target


def _compile(
    circuits: QuantumCircuit | Iterable[QuantumCircuit] = None,
    *args,
    qubit_labels: Sequence[int] | None = None,
    target: Target | None = None,
    verbatim: bool | None = None,
    basis_gates: Collection[str] | None = None,
    coupling_map: list[list[int]] | None = None,
    angle_restrictions: Mapping[str, Mapping[int, set[float] | tuple[float, float]]] | None = None,
    optimization_level: int = 0,
    callback: Callable | None = None,
    num_processes: int | None = None,
    pass_manager: PassManager | None = None,
    braket_device: Device | None = None,
    connectivity: list[list[int]] | None = None,
    verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME,
    layout_method: str | None = None,
    routing_method: str | None = None,
    seed_transpiler: int | None = None,
) -> _CompilationContext:
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    circuits = list(circuits)

    if len(args) > 4:
        raise ValueError(f"Unknown arguments passed: {args[4:]}")
    padded = args + (None,) * max(0, 4 - len(args))
    basis_gates = _check_positional(padded[0], basis_gates, "basis_gates")
    verbatim = _check_positional(padded[1], verbatim, "verbatim")
    connectivity = _check_positional(padded[2], connectivity, "connectivity")
    angle_restrictions = _check_positional(padded[3], angle_restrictions, "angle_restrictions")
    _validate_arguments(
        circuits, target, basis_gates, coupling_map, connectivity, pass_manager, braket_device
    )
    coupling_map = coupling_map or connectivity

    has_verbatim_boxes = any(
        isinstance(instr.operation, BoxOp)
        and getattr(instr.operation, "label", None) == verbatim_box_name
        for circ in circuits
        for instr in circ.data
    )

    if any(
        isinstance(instr.operation, Barrier)
        and getattr(instr.operation, "label", None) == verbatim_box_name
        for circ in circuits
        for instr in circ.data
    ):
        raise ValueError(
            "Cannot have a Barrier labeled with the same label used for verbatim boxes"
        )

    if pass_manager and has_verbatim_boxes:
        raise ValueError(
            "Custom pass_manager is not supported with verbatim boxes. "
            "Verbatim boxes require controlled transpilation to preserve gate ordering."
        )

    if braket_device:
        if qubit_labels:
            raise ValueError("Cannot specify qubit labels with Braket device")
        target = (
            aws_device_to_target(braket_device)
            if isinstance(braket_device, AwsDevice)
            else local_simulator_to_target(braket_device)
        )
        qubit_labels = (
            tuple(sorted(braket_device.topology_graph.nodes))
            if isinstance(braket_device, AwsDevice) and braket_device.topology_graph
            else None
        )

    if pass_manager:
        circuits = pass_manager.run(circuits, callback=callback, num_processes=num_processes)
    elif not verbatim:
        target = target if basis_gates or coupling_map or target else _default_target(circuits)

        if has_verbatim_boxes:
            warnings.warn(
                "Overriding layout method to 'trivial' "
                "and routing method to 'none' as the circuit has verbatim blocks",
                stacklevel=1,
            )
            effective_layout_method = "trivial"
            effective_routing_method = "none"
        else:
            effective_layout_method = layout_method
            effective_routing_method = routing_method

        if (
            target
            or coupling_map
            or (
                basis_gates
                and not {instr.operation.name for circ in circuits for instr in circ.data}.issubset(
                    basis_gates
                )
            )
        ):
            pm = generate_preset_pass_manager(
                # generate_preset_pass_manager does not accept None unlike transpile().
                # If user explicitly passes None, default to 2 to match Qiskit's transpile() behavior.
                # When not passed, the signature default of 0 is used (no optimization).
                optimization_level=optimization_level if optimization_level is not None else 2,
                basis_gates=list(basis_gates) if basis_gates else None,
                coupling_map=coupling_map,
                target=target,
                layout_method=effective_layout_method,
                routing_method=effective_routing_method,
                seed_transpiler=seed_transpiler,
            )
            if has_verbatim_boxes:
                pm.pre_init = PassManager([ExtractVerbatimBoxes(verbatim_box_name)])
                pm.post_optimization = PassManager([RestoreVerbatimBoxes(verbatim_box_name)])
            circuits = pm.run(circuits, callback=callback, num_processes=num_processes)
        elif has_verbatim_boxes:
            # No transpilation needed but still need to extract/restore verbatim boxes
            verbatim_pm = PassManager([
                ExtractVerbatimBoxes(verbatim_box_name),
                RestoreVerbatimBoxes(verbatim_box_name),
            ])
            circuits = verbatim_pm.run(circuits)
    elif has_verbatim_boxes:
        # verbatim=True: unpack BoxOps without transpilation
        verbatim_pm = PassManager([
            ExtractVerbatimBoxes(verbatim_box_name),
            RestoreVerbatimBoxes(verbatim_box_name),
        ])
        circuits = verbatim_pm.run(circuits)

    if isinstance(target, _SubstitutedTarget):
        circuits = target._substitute(circuits)

    return _CompilationContext(
        circuits=circuits,
        target=target,
        qubit_labels=qubit_labels,
        verbatim=verbatim,
        basis_gates=basis_gates,
        angle_restrictions=angle_restrictions,
        pass_manager=pass_manager,
    )


def _check_positional(pos: _T, kw: _T, name: str) -> _T:
    if pos is None:
        return kw
    if kw is not None:
        raise TypeError(f"Multiple values for {name}: {pos, kw}")
    warnings.warn(
        f"Passing {name} as a positional argument is deprecated.",
        DeprecationWarning,
        stacklevel=1,
    )
    return pos


def _validate_arguments(
    circuits: list[QuantumCircuit],
    target: Target | None,
    basis_gates: Collection[str] | None,
    coupling_map: list[list[int]] | None,
    connectivity: list[list[int]] | None,
    pass_manager: PassManager | None,
    braket_device: Device | None,
) -> None:
    if other_types := {type(c).__name__ for c in circuits if not isinstance(c, QuantumCircuit)}:
        raise TypeError(f"Expected only QuantumCircuits, got {other_types} instead.")
    if connectivity:
        if coupling_map:
            raise ValueError("Cannot specify both coupling_map and connectivity")
        warnings.warn(
            "connectivity is deprecated; use coupling_map instead.",
            DeprecationWarning,
            stacklevel=1,
        )
    if (
        sum([
            (1 if target else 0),
            (1 if (basis_gates or coupling_map or connectivity) else 0),
            (1 if pass_manager else 0),
            (1 if braket_device else 0),
        ])
        > 1
    ):
        raise ValueError(
            "Cannot only specify one of {target, (basis_gates or coupling map/connectivity), "
            "pass_manager, braket_device}"
        )
