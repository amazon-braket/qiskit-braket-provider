"""Compilation pipeline for Qiskit circuits targeting Braket devices.

This module contains the internal compilation pipeline used by to_braket()
and the run() endpoints, including verbatim box handling and transpilation.
"""

import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Barrier, BoxOp, Measure
from qiskit.transpiler import PassManager, Target

from braket.aws import AwsDevice
from braket.devices import Device
from qiskit_braket_provider.providers.gate_mappings import (
    _BRAKET_GATE_NAME_TO_QISKIT_GATE,
    _BRAKET_TO_QISKIT_NAMES,
    _BRAKET_VERBATIM_BOX_NAME,
)
from qiskit_braket_provider.providers.target import (
    SubstitutedTarget,
    aws_device_to_target,
    local_simulator_to_target,
)

_T = TypeVar("_T")


def _extract_verbatim_boxes(
    circuit: QuantumCircuit, verbatim_box_name: str
) -> tuple[QuantumCircuit, list[tuple[QuantumCircuit, list[int]]]]:
    """Extract BoxOp operations with verbatim box name and replace with barriers.

    Args:
        circuit: The Qiskit circuit to process
        verbatim_box_name: The label name used to identify verbatim BoxOp operations

    Returns:
        A tuple of (modified_circuit, verbatim_boxes) where:
        - modified_circuit: Circuit with BoxOps replaced by named barriers
        - verbatim_boxes: List of (box_circuit, qubit_indices) tuples
    """
    modified_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    modified_circuit.global_phase = circuit.global_phase

    verbatim_boxes = []

    for instruction in circuit.data:
        operation = instruction.operation

        qubit_indices = [circuit.find_bit(q).index for q in instruction.qubits]
        clbit_indices = [circuit.find_bit(q).index for q in instruction.clbits]

        if isinstance(operation, BoxOp) and getattr(operation, "label", None) == verbatim_box_name:
            box_circuit = operation.blocks[0]
            verbatim_boxes.append((box_circuit, qubit_indices))
            barrier = Barrier(len(instruction.qubits), label=verbatim_box_name)
            modified_circuit.append(barrier, qubit_indices, clbit_indices)
        else:
            modified_circuit.append(operation, qubit_indices, clbit_indices)

    return modified_circuit, verbatim_boxes


def _restore_verbatim_boxes(
    transpiled_circuit: QuantumCircuit,
    verbatim_boxes: list[tuple[QuantumCircuit, list[int]]],
    verbatim_box_name: str,
) -> QuantumCircuit:
    """Restore verbatim boxes by replacing named barriers with box contents.

    Args:
        transpiled_circuit: The transpiled circuit with named barriers
        verbatim_boxes: List of (box_circuit, original_qubit_indices) tuples
        verbatim_box_name: The label name used to identify verbatim barriers

    Returns:
        Circuit with verbatim box contents restored

    Raises:
        ValueError: If barrier count doesn't match verbatim box count
        ValueError: If qubit mapping fails
    """
    reconstructed_circuit = transpiled_circuit.copy_empty_like()

    verbatim_box_iter = iter(verbatim_boxes)
    barrier_count = 0

    for instruction in transpiled_circuit.data:
        operation = instruction.operation

        if (
            isinstance(operation, Barrier)
            and getattr(operation, "label", None) == verbatim_box_name
        ):
            barrier_count += 1

            try:
                box_circuit, _ = next(verbatim_box_iter)
            except StopIteration as err:
                raise ValueError(
                    f"Compiler error while processing verbatim boxes. Illegal barriers with label '{verbatim_box_name}'"
                ) from err

            for box_instruction in box_circuit.data:
                qubit_indices = [box_circuit.find_bit(q).index for q in box_instruction.qubits]
                clbit_indices = [box_circuit.find_bit(q).index for q in box_instruction.clbits]
                reconstructed_circuit.append(
                    box_instruction.operation, qubit_indices, clbit_indices
                )
        else:
            qubit_indices = [transpiled_circuit.find_bit(q).index for q in instruction.qubits]
            clbit_indices = [transpiled_circuit.find_bit(q).index for q in instruction.clbits]
            reconstructed_circuit.append(operation, qubit_indices, clbit_indices)

    remaining_boxes = list(verbatim_box_iter)
    if remaining_boxes:
        raise ValueError(
            f"Compiler error while processing verbatim boxes. Expected {barrier_count} "
            f"verbatim boxes, but found {len(verbatim_boxes)}."
        )

    return reconstructed_circuit


@dataclass(frozen=True)
class CompilationContext:
    """Result of :func:`compile_circuits` containing compiled circuits and resolved state.

    Attributes:
        circuits: The compiled Qiskit QuantumCircuits.
        target: The resolved transpiler target, if any.
        qubit_labels: Physical qubit indices on the target device.
        verbatim: Whether verbatim mode was requested.
        basis_gates: The basis gate set used for compilation.
        angle_restrictions: Per-gate angle constraints from the device.
        pass_manager: The custom PassManager used, if any.
    """

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


def compile_circuits(
    circuits: QuantumCircuit | Iterable[QuantumCircuit],
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
) -> CompilationContext:
    """Compile Qiskit circuits for execution on a Braket device.

    This is the compilation pipeline used by :func:`to_braket`. It handles
    verbatim box extraction/restoration, transpilation via Qiskit's pass infrastructure,
    and device-specific gate substitutions.

    Args:
        circuits: One or more Qiskit QuantumCircuits to compile.
        qubit_labels: Physical qubit indices on the target device. If not supplied,
            contiguous indices are assumed.
        target: A Qiskit transpiler target describing device constraints.
        verbatim: If ``True``, skip transpilation (pass circuits through unchanged).
        basis_gates: Set of gate names supported by the target device.
        coupling_map: Qubit connectivity as a list of ``[control, target]`` pairs.
        angle_restrictions: Per-gate angle constraints from the device.
        optimization_level: Transpiler optimization level (0-3). Default: 0.
        callback: Callback function passed to the transpiler.
        num_processes: Number of parallel processes for transpilation.
        pass_manager: A custom Qiskit PassManager. Mutually exclusive with other
            transpilation options.
        braket_device: A Braket Device to derive target and qubit labels from.
        connectivity: Deprecated alias for ``coupling_map``.
        verbatim_box_name: Label identifying verbatim BoxOp nodes.
        layout_method: Layout method for the transpiler.
        routing_method: Routing method for the transpiler.
        seed_transpiler: Seed for reproducible transpilation.

    Returns:
        A :class:`CompilationContext` containing the compiled circuits and resolved
        compilation state.

    Raises:
        ValueError: If mutually exclusive options are specified together.
        TypeError: If inputs are not QuantumCircuits.
    """
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

    has_barriers_named_verbatim = False
    has_verbatim_boxes = False

    for circ in circuits:
        for instr in circ.data:
            label = getattr(instr.operation, "label", None)
            if label == verbatim_box_name:
                if isinstance(instr.operation, Barrier):
                    has_barriers_named_verbatim = True
                elif isinstance(instr.operation, BoxOp):
                    has_verbatim_boxes = True

    if has_barriers_named_verbatim:
        raise ValueError(
            "Cannot have a Barrier labeled with the same label used for verbatim boxes"
        )

    if pass_manager and has_verbatim_boxes:
        raise ValueError(
            "Custom pass_manager is not supported with verbatim boxes. "
            "Verbatim boxes require controlled transpilation to preserve gate ordering."
        )

    all_verbatim_boxes = []
    if has_verbatim_boxes:
        extracted_circuits = []
        for circ in circuits:
            modified_circ, verbatim_boxes = _extract_verbatim_boxes(circ, verbatim_box_name)
            extracted_circuits.append(modified_circ)
            all_verbatim_boxes.append(verbatim_boxes)
        circuits = extracted_circuits

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
            circuits = transpile(
                circuits,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                optimization_level=optimization_level,
                target=target,
                callback=callback,
                num_processes=num_processes,
                layout_method=effective_layout_method,
                routing_method=effective_routing_method,
                seed_transpiler=seed_transpiler,
            )
    if isinstance(target, SubstitutedTarget):
        circuits = target._substitute(circuits)

    if has_verbatim_boxes:
        circuits = [
            _restore_verbatim_boxes(circ, verbatim_boxes, verbatim_box_name)
            if len(verbatim_boxes) > 0
            else circ
            for circ, verbatim_boxes in zip(circuits, all_verbatim_boxes, strict=False)
        ]

    return CompilationContext(
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
