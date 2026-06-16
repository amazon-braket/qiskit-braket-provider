"""Util function for provider.

This module provides utilities for converting between Braket and Qiskit quantum circuits,
including support for Braket verbatim pragmas. Verbatim boxes in OpenQASM 3 programs are
converted to Qiskit BoxOp operations, which treat blocks of gates atomically to preserve
sequences that should not be optimized.
"""

import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, TypeVar, overload

import numpy as np
import qiskit.circuit.library as qiskit_gates
import qiskit.quantum_info as qiskit_qi
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import (
    Barrier,
    BoxOp,
    ControlledGate,
    Measure,
    Parameter,
    ParameterExpression,
    ParameterVectorElement,
)
from qiskit.circuit import Instruction as QiskitInstruction
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.transpiler import (
    PassManager,
    Target,
)
from qiskit_ionq import add_equivalences

from braket.aws import AwsDevice
from braket.circuits import Circuit, Instruction, measure
from braket.circuits import Observable as BraketObservable
from braket.circuits import observables as braket_observables
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.devices import Device
from braket.ir.openqasm import Program
from braket.parametric import FreeParameter, FreeParameterExpression, Parameterizable
from qiskit_braket_provider.providers.constants import (
    _BRAKET_GATE_NAME_TO_QISKIT_GATE,
    _BRAKET_SUPPORTED_NOISES,
    _BRAKET_TO_QISKIT_NAMES,
    _BRAKET_VERBATIM_BOX_NAME,
    _CONTROLLED_GATES_BY_QUBIT_COUNT,  # noqa: F401
    _EPS,
    _PAULI_MAP,
    _QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES,
    _QISKIT_GATE_NAME_TO_BRAKET_GATE,
    _TRANSPILER_GATE_SUBSTITUTES,  # noqa: F401
)
from qiskit_braket_provider.providers.qasm_context import (
    QiskitProgramContext,
    sympy_to_qiskit,
)
from qiskit_braket_provider.providers.target import (
    SubstitutedTarget,
    aws_device_to_target,
    gateset_from_properties,  # noqa: F401
    get_controlled_gateset,  # noqa: F401
    local_simulator_to_target,
    native_angle_restrictions,  # noqa: F401
    native_gate_connectivity,  # noqa: F401
    native_gate_set,  # noqa: F401
)

add_equivalences()

_Translatable: TypeAlias = QuantumCircuit | Circuit | Program | str
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

        # Convert Qubit objects to integer indices
        # instruction.qubits contains Qubit objects (circuit-specific)
        # find_bit(q).index returns the global integer index (0, 1, 2, ...)
        # We consistently use indices for circuits with physical qubits as they do not go through mapping and routing
        qubit_indices = [circuit.find_bit(q).index for q in instruction.qubits]
        clbit_indices = [circuit.find_bit(q).index for q in instruction.clbits]

        if isinstance(operation, BoxOp) and getattr(operation, "label", None) == verbatim_box_name:
            # Extract the circuit from the BoxOp (first block)
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

            # Insert gates from the verbatim box directly (not as BoxOp)
            # Since verbatim boxes can only exist in circuits using physical qubits,
            # and we use trivial layout (identity mapping) with no routing during transpilation,
            # the qubit indices remain unchanged between the box circuit and the reconstructed circuit.
            for box_instruction in box_circuit.data:
                qubit_indices = [box_circuit.find_bit(q).index for q in box_instruction.qubits]
                clbit_indices = [box_circuit.find_bit(q).index for q in box_instruction.clbits]
                # Append the gate instruction with the same qubits as in the box
                reconstructed_circuit.append(
                    box_instruction.operation, qubit_indices, clbit_indices
                )
        else:
            # Get indices of qubits and clbits and add instruction as-is
            qubit_indices = [transpiled_circuit.find_bit(q).index for q in instruction.qubits]
            clbit_indices = [transpiled_circuit.find_bit(q).index for q in instruction.clbits]
            reconstructed_circuit.append(operation, qubit_indices, clbit_indices)

    remaining_boxes = list(verbatim_box_iter)
    if remaining_boxes:
        raise ValueError(
            f"Compiler error while processing verbatim boxes. Expected {barrier_count} "
            "verbatim boxes, but found {len(verbatim_boxes)}."
        )

    return reconstructed_circuit


@dataclass(frozen=True)
class _CompilationContext:
    """Internal result from _compile containing compiled circuits and resolved state."""

    circuits: list[QuantumCircuit]
    single_instance: bool
    target: Target | None
    qubit_labels: Sequence[int] | None
    verbatim: bool | None
    basis_gates: Collection[str] | None
    angle_restrictions: Mapping[str, Mapping[int, set[float] | tuple[float, float]]] | None
    pass_manager: PassManager | None


def _compile(
    circuits: _Translatable | Iterable[_Translatable] = None,
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
    add_measurements: bool = True,
    circuit: _Translatable | Iterable[_Translatable] | None = None,
    connectivity: list[list[int]] | None = None,
    verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME,
    layout_method: str | None = None,
    routing_method: str | None = None,
    seed_transpiler: int | None = None,
) -> _CompilationContext:

    circuits, single_instance = _get_circuits(circuits, circuit, add_measurements)
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
                # Check if any circuits have barriers labeled the same as a verbatim box, and if so raise an error
                if isinstance(instr.operation, Barrier):
                    has_barriers_named_verbatim = True
                # Check if any circuits have verbatim boxes and extract them before transpilation
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

    return _CompilationContext(
        circuits=circuits,
        single_instance=single_instance,
        target=target,
        qubit_labels=qubit_labels,
        verbatim=verbatim,
        basis_gates=basis_gates,
        angle_restrictions=angle_restrictions,
        pass_manager=pass_manager,
    )


@overload
def to_braket(
    circuits: _Translatable = ...,
    *args,
    **kwargs,
) -> Circuit: ...


@overload
def to_braket(
    circuits: Iterable[_Translatable],
    *args,
    **kwargs,
) -> list[Circuit]: ...


def to_braket(
    circuits: _Translatable | Iterable[_Translatable] = None,
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
    add_measurements: bool = True,
    circuit: _Translatable | Iterable[_Translatable] | None = None,
    connectivity: list[list[int]] | None = None,
    verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME,
    layout_method: str | None = None,
    routing_method: str | None = None,
    seed_transpiler: int | None = None,
) -> Circuit | list[Circuit]:
    """Converts a single or list of Qiskit QuantumCircuits to a single or list of Braket Circuits.

    The recommended way to use this method is to minimally pass in qubit labels and a target
    (instead of basis gates and coupling map). This ensures that the translated circuit is actually
    supported by the device (and doesn't, for example, include unsupported parameters for gates).
    The latter guarantees that the output Braket circuit uses the qubit labels of the Braket device,
    which are not necessarily contiguous.

    Args:
        circuits (QuantumCircuit | Circuit | Program | str | Iterable): Qiskit or Braket
            circuit(s) or OpenQASM 3 program(s) to transpile and translate to Braket.
        qubit_labels (Sequence[int] | None): A list of (not necessarily contiguous) indices of
            qubits in the underlying Amazon Braket device. If not supplied, then the indices are
            assumed to be contiguous. Default: ``None``.
        target (Target | None): A backend transpiler target. Can only be provided
            if basis_gates is ``None``. Default: ``None``.
        verbatim (bool): Whether to translate the circuit without any modification, in other
            words without transpiling it. Default: ``False``.
        basis_gates (Collection[str] | None): The gateset to transpile to. Can only be provided
            if target is ``None``. If ``None`` and target is ``None``, the transpiler will use
            all gates defined in the Braket SDK. Default: ``None``.
        coupling_map (list[list[int]] | None): If provided, will transpile to a circuit
            with this coupling map (reflects Qiskit physical qubits). Default: ``None``.
        angle_restrictions (Mapping[str, Mapping[int, set[float] | tuple[float, float]]] | None):
            Mapping of gate names to parameter angle constraints used to
            validate numeric parameters. Default: ``None``.
        optimization_level (int | None): The optimization level to pass to ``qiskit.transpile``.
            From Qiskit:

            * 0: no optimization - basic translation, no optimization, trivial layout
            * 1: light optimization - routing + potential SaberSwap, some gate cancellation
              and 1Q gate folding
            * 2: medium optimization - better routing (noise aware) and commutative cancellation
            * 3: high optimization - gate resynthesis and unitary-breaking passes

            Default: 0.
        callback (Callable | None): A callback function that will be called after each transpiler
            pass execution. Default: ``None``.
        num_processes (int | None): The maximum number of parallel transpilation processes for
            multiple circuits. Default: ``None``.
        pass_manager (PassManager): `PassManager` to transpile the circuit; will raise an error if
            used in conjunction with a target, basis gates, or connectivity. Default: ``None``.
        braket_device (Device): Braket device to transpile to. Can only be provided if `target`
            and ``basis_gates`` are ``None``. Default: ``None``.
        add_measurements (bool): Whether to add measurements when translating Braket circuits.
            Default: True.
        circuit (QuantumCircuit | Circuit | Program | str | Iterable | None): Qiskit or Braket
            circuit(s) or OpenQASM 3 program(s) to transpile and translate to Braket.
            Default: ``None``. DEPRECATED: use first positional argument or ``circuits`` instead.
        connectivity (list[list[int]] | None): If provided, will transpile to a circuit
            with this connectivity. Default: ``None``. DEPRECATED: use ``coupling_map`` instead.
        verbatim_box_name (str): The label name used to identify verbatim BoxOp operations
            in Qiskit circuits. When circuits contain BoxOp operations with this label, they
            will be preserved during transpilation by temporarily replacing them with barriers.
            Default: ``"verbatim"``.
        layout_method (str | None): The layout method to use during transpilation. If ``None``
            and the circuit contains verbatim boxes, defaults to ``'trivial'`` to preserve
            physical qubit mappings. Otherwise uses Qiskit's default. Default: ``None``.
        routing_method (str | None): The routing method to use during transpilation. If ``None``
            and the circuit contains verbatim boxes, defaults to ``'none'`` to disable routing
            and preserve physical qubit structure. Otherwise uses Qiskit's default. Default: ``None``.
        seed_transpiler (int | None): This specifies a seed used for the stochastic parts
            of the transpiler. Default: ``None``.

    Raises:
        ValueError: If more than one of `target`, ``basis_gates``
            or ``coupling_map``/``connectivity``, ``pass_manager``, and ``braket_device``
            are passed together, or if `qubit_labels` is passed with ``braket_device``.

    Returns:
        Circuit | list[Circuit]: Braket circuit or circuits
    """
    result = _compile(
        circuits,
        *args,
        qubit_labels=qubit_labels,
        target=target,
        verbatim=verbatim,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        angle_restrictions=angle_restrictions,
        optimization_level=optimization_level,
        callback=callback,
        num_processes=num_processes,
        pass_manager=pass_manager,
        braket_device=braket_device,
        add_measurements=add_measurements,
        circuit=circuit,
        connectivity=connectivity,
        verbatim_box_name=verbatim_box_name,
        layout_method=layout_method,
        routing_method=routing_method,
        seed_transpiler=seed_transpiler,
    )
    translated = [
        _translate_to_braket(
            circ,
            result.target,
            result.qubit_labels,
            result.verbatim,
            result.basis_gates,
            result.angle_restrictions,
            result.pass_manager,
        )
        for circ in result.circuits
    ]
    return translated[0] if result.single_instance else translated


def _get_circuits(
    circuits: _Translatable | Iterable[_Translatable] | None,
    circuit: _Translatable | Iterable[_Translatable] | None,
    add_measurements: bool,
) -> tuple[list[QuantumCircuit], bool]:
    if circuit is not None and circuits is not None:
        raise ValueError("Cannot specify both circuits and circuit")
    if circuit is None and circuits is None:
        raise ValueError("Must specify circuits to transpile")
    if circuit is not None:
        warnings.warn(
            "circuit is deprecated; use circuits instead.", DeprecationWarning, stacklevel=1
        )
        circuits = circuit
    single_instance = isinstance(circuits, _Translatable) or not isinstance(circuits, Iterable)
    if single_instance:
        circuits = [circuits]
    return [
        to_qiskit(c, add_measurements=add_measurements)
        if isinstance(c, (Circuit, Program, str))
        else c
        for c in circuits
    ], single_instance


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


def _translate_to_braket(
    circuit: QuantumCircuit,
    target: Target | None,
    qubit_labels: Sequence[int] | None,
    verbatim: bool,
    basis_gates: Iterable[str] | None,
    angle_restrictions: Mapping[str, Mapping[int, set[float] | tuple[float, float]]] | None,
    pass_manager: PassManager | None,
) -> Circuit:
    # Verify that ParameterVector would not collide with scalar variables after renaming.
    _validate_name_conflicts(circuit.parameters)
    # Handle qiskit to braket conversion
    measured_qubits: dict[int, int] = {}
    braket_circuit = Circuit()
    qubit_labels = qubit_labels or _default_qubit_labels(circuit)
    for circuit_instruction in circuit.data:
        operation = circuit_instruction.operation
        qubits = circuit_instruction.qubits

        if getattr(operation, "condition", None):
            raise NotImplementedError(
                "Conditional operations are not supported. "
                f"Found conditional gate '{operation.name}'. "
                f"Only MeasureFF and CCPRx gates are supported in Braket."
            )

        match gate_name := operation.name:
            case "measure":
                qubit = qubits[0]  # qubit count = 1 for measure
                qubit_index = qubit_labels[circuit.find_bit(qubit).index]
                if qubit_index in measured_qubits.values():
                    raise ValueError(f"Cannot measure previously measured qubit {qubit_index}")
                clbit = circuit.find_bit(circuit_instruction.clbits[0]).index
                measured_qubits[clbit] = qubit_index
            case "barrier":
                if target and "barrier" in target.operation_names:
                    qubit_indices = [
                        qubit_labels[circuit.find_bit(qubit).index] for qubit in qubits
                    ]
                    braket_circuit.barrier(target=qubit_indices or None)
                else:
                    warnings.warn(
                        "Barrier is not included in the current Target and will be ignored.",
                        stacklevel=2,
                    )
            case "reset":
                raise NotImplementedError(
                    "reset operation not supported by qiskit to braket adapter"
                )
            case "unitary" | "kraus":
                params = _create_free_parameters(operation)
                qubit_indices = [qubit_labels[circuit.find_bit(qubit).index] for qubit in qubits][
                    ::-1
                ]  # reversal for little to big endian notation

                for gate in _QISKIT_GATE_NAME_TO_BRAKET_GATE[gate_name](params):
                    braket_circuit += Instruction(
                        operator=gate,
                        target=qubit_indices,
                    )
            case _:
                if (
                    isinstance(operation, ControlledGate)
                    and operation.ctrl_state != 2**operation.num_ctrl_qubits - 1
                ):
                    raise ValueError("Negative control is not supported")
                # Getting the index from the bit mapping
                qubit_indices = [qubit_labels[circuit.find_bit(qubit).index] for qubit in qubits]
                if intersection := set(measured_qubits.values()).intersection(qubit_indices):
                    raise ValueError(
                        f"Cannot apply operation {gate_name} to measured qubits {intersection}"
                    )
                params = _create_free_parameters(operation)
                # TODO: Use angle_bounds in Target.add_instruction instead of validating here
                _validate_angle_restrictions(gate_name, params, angle_restrictions)
                if gate_name in _QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES:
                    for gate in _QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES[gate_name](*params):
                        gate_qubit_count = gate.qubit_count
                        braket_circuit += Instruction(
                            operator=gate,
                            target=qubit_indices[-gate_qubit_count:],
                            control=qubit_indices[:-gate_qubit_count],
                        )
                else:
                    for gate in _QISKIT_GATE_NAME_TO_BRAKET_GATE[gate_name](*params):
                        braket_circuit += Instruction(
                            operator=gate,
                            target=qubit_indices,
                        )
    global_phase = circuit.global_phase
    has_nonzero_phase = isinstance(global_phase, ParameterExpression) or abs(global_phase) > _EPS
    if has_nonzero_phase:
        if (target and "global_phase" in target) or (basis_gates and "global_phase" in basis_gates):
            if isinstance(global_phase, ParameterExpression):
                global_phase = FreeParameterExpression(rename_parameter(global_phase))
            braket_circuit.gphase(global_phase)
        else:
            warnings.warn(
                f"Device does not support global phase; "
                f"global phase of {global_phase} will not be included in Braket circuit",
                stacklevel=2,
            )

    # QPU targets will have qubits/pairs specified for each instruction;
    # Targets whose values consist solely of {None: None} are either simulator or default targets
    if verbatim or (target and any(v != {None: None} for v in target.values())) or pass_manager:
        braket_circuit = Circuit(braket_circuit.result_types).add_verbatim_box(
            Circuit(braket_circuit.instructions)
        )

    for clbit in sorted(measured_qubits):
        braket_circuit.measure(measured_qubits[clbit])

    return braket_circuit


def _default_target(circuits: Iterable[QuantumCircuit]) -> Target:
    num_qubits = max(circuit.num_qubits for circuit in circuits)
    target = Target(num_qubits=num_qubits)
    for braket_name, instruction in _BRAKET_GATE_NAME_TO_QISKIT_GATE.items():
        if name := _BRAKET_TO_QISKIT_NAMES.get(braket_name.lower()):
            target.add_instruction(instruction, name=name)
    target.add_instruction(Measure())
    target.add_instruction(Barrier(1))
    return target


def _default_qubit_labels(circuit: QuantumCircuit) -> tuple[int, ...]:
    bits = sorted(circuit.find_bit(q).index for q in circuit.qubits)
    return tuple(range(max(bits) + 1)) if bits else ()


def _create_free_parameters(operation: QiskitInstruction) -> list[Any]:
    for i, param in enumerate(params := operation.params):
        match param:
            case Parameter() | ParameterVectorElement():
                params[i] = FreeParameter(rename_parameter(param))
            case ParameterExpression():
                params[i] = FreeParameterExpression(rename_parameter(param))
    return params


def _validate_angle_restrictions(
    gate_name: str,
    params: Iterable,
    angle_restrictions: Mapping[str, Mapping[int, set[float] | tuple[float, float]]] | None,
) -> None:
    """Validate gate parameter angles against a restriction map.

    Parameters that are ``FreeParameter`` or ``ParameterExpression`` instances
    are ignored. Numeric angles are validated against the entry in
    ``angle_restrictions`` for the ``gate_name``. Each restriction can be a set
    of discrete allowed values or a ``(min, max)`` tuple describing an inclusive
    range.
    """
    if not angle_restrictions or gate_name not in angle_restrictions:
        return
    restrictions = angle_restrictions[gate_name]
    params = list(params)
    for index, restriction in restrictions.items():
        if index >= len(params):
            continue
        param = params[index]
        if isinstance(
            param,
            (
                FreeParameter,
                FreeParameterExpression,
                ParameterExpression,
            ),
        ):
            continue
        angle = float(param)
        if isinstance(restriction, set):
            if not any(abs(angle - allowed) <= _EPS for allowed in restriction):
                raise ValueError(
                    f"Angle {angle} for {gate_name} parameter {index} is not supported"
                )
        else:
            min_angle, max_angle = restriction
            if angle < min_angle - _EPS or angle > max_angle + _EPS:
                raise ValueError(
                    f"Angle {angle} for {gate_name} parameter {index} "
                    f"not in range [{min_angle}, {max_angle}]"
                )


def rename_parameter(parameter: Parameter) -> str:
    """Translates a parameter in a ParameterVector to a Braket-compatible parameter name.

    Args:
        parameter (Parameter): The Qiskit parameter to translate.

    Returns:
        str: The Braket-compatible parameter name.
    """
    return str(parameter).replace("[", "_").replace("]", "")


def _validate_name_conflicts(parameters: Collection[Parameter]) -> None:
    renamed_parameters = {rename_parameter(param) for param in parameters}
    if len(renamed_parameters) != len(parameters):
        raise ValueError(
            "ParameterVector elements are renamed from v[i] to v_i, which resulted "
            "in a conflict with another parameter. Please rename your parameters."
        )


def translate_sparse_pauli_op(op: SparsePauliOp) -> BraketObservable:
    """
    Translate a SparsePauliOp to a Braket observable.

    Args:
        op (SparsePauliOp): Operation to translate.

    Returns:
        BraketObservable: Corresponding Braket observable.
    """
    return (
        braket_observables.Sum([
            _translate_pauli(pauli, np.real(coeff))
            for pauli, coeff in zip(op.paulis, op.coeffs, strict=True)
        ])
        if len(op) > 1
        else _translate_pauli(op.paulis[0], np.real(op.coeffs[0]))
    )


def _translate_pauli(pauli: Pauli, coeff: float = 1.0) -> BraketObservable:
    """
    Translate a single Pauli and a coefficient to a Braket observable.

    Args:
        pauli (Pauli): Pauli observable to translate.
        coeff (float): Coefficient of the Pauli. Default: 1.

    Returns:
        BraketObservable: Corresponding Braket observable.
    """
    factors = [
        _PAULI_MAP[pauli_char](i)
        for i, pauli_char in enumerate(reversed(str(pauli)))
        if pauli_char != "I"
    ]
    if not factors:
        return (
            braket_observables.I(0) * coeff
        )  # Still include trivial term so expectation is correct
    return (braket_observables.TensorProduct(factors) if len(factors) > 1 else factors[0]) * coeff


def to_qiskit(
    circuit: Circuit | Program | str,
    add_measurements: bool = True,
    verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME,
) -> QuantumCircuit:
    """Return a Qiskit quantum circuit from a Braket quantum circuit.

    Args:
        circuit (Circuit | Program | str): Braket quantum circuit or OpenQASM 3 program.
        add_measurements (bool): Whether to append measurements in the conversion
        verbatim_box_name (str): Name to use for BoxOp labels when converting verbatim boxes.
            Default: "verbatim"

    Returns:
        QuantumCircuit: Qiskit quantum circuit

    Examples:
        Convert an OpenQASM 3 program with a verbatim box:

        >>> openqasm_program = '''
        ... OPENQASM 3.0;
        ... #pragma braket verbatim
        ... box {
        ...     h $0;
        ...     cnot $0, $1;
        ... }
        ... '''
        >>> qiskit_circuit = to_qiskit(openqasm_program)
        >>> # The verbatim box is represented as a BoxOp in the circuit
        >>> # You can inspect it by iterating through the circuit operations
        >>> for instruction in qiskit_circuit.data:
        ...     if hasattr(instruction.operation, 'label') and instruction.operation.label == 'verbatim':
        ...         print(f"Found verbatim box: {instruction.operation}")

        Use a custom name for verbatim boxes:

        >>> qiskit_circuit = to_qiskit(openqasm_program, verbatim_box_name="my_verbatim")
        >>> # All verbatim boxes will have the label "my_verbatim"
    """
    if isinstance(circuit, Program):
        return (
            Interpreter(QiskitProgramContext(verbatim_box_name))
            .run(circuit.source, inputs=circuit.inputs)
            .circuit
        )
    if isinstance(circuit, str):
        return Interpreter(QiskitProgramContext(verbatim_box_name)).run(circuit).circuit
    if not isinstance(circuit, Circuit):
        raise TypeError(f"Expected a Circuit, got {type(circuit)} instead.")

    num_measurements = sum(
        isinstance(instr.operator, measure.Measure) for instr in circuit.instructions
    )
    qiskit_circuit = QuantumCircuit(circuit.qubit_count, num_measurements)
    qubit_map = {int(qubit): index for index, qubit in enumerate(sorted(circuit.qubits))}
    parameter_map: dict[str, Parameter] = {}
    cbit = 0
    for instruction in circuit.instructions:
        operator = instruction.operator
        gate_name = operator.name.lower()

        # Handle barrier separately
        if gate_name == "barrier":
            barrier_qubits = [qiskit_circuit.qubits[qubit_map[i]] for i in instruction.target]
            qiskit_circuit.barrier(barrier_qubits)
            continue

        if gate_name in _BRAKET_SUPPORTED_NOISES:
            gate = _create_qiskit_kraus(operator.to_matrix())
        elif gate_name == "unitary":
            gate = _create_qiskit_unitary(operator.to_matrix())
        else:
            gate = _create_qiskit_gate(
                gate_name,
                (operator.parameters if isinstance(operator, Parameterizable) else []),
                parameter_map,
            )
        if (power := instruction.power) != 1:
            gate = gate**power
        if control_qubits := instruction.control:
            ctrl_state = instruction.control_state.as_string[::-1]
            gate = gate.control(len(control_qubits), ctrl_state=ctrl_state)

        target = [qiskit_circuit.qubits[qubit_map[i]] for i in control_qubits]
        target += [qiskit_circuit.qubits[qubit_map[i]] for i in instruction.target]

        if gate_name == "measure":
            qiskit_circuit.append(gate, target, [cbit])
            cbit += 1
        else:
            qiskit_circuit.append(gate, target)
    if num_measurements == 0 and add_measurements:
        qiskit_circuit.measure_all()
    return qiskit_circuit


def _create_qiskit_unitary(matrix: np.ndarray) -> qiskit_gates.UnitaryGate:
    return qiskit_gates.UnitaryGate(_reverse_endianness(matrix))


def _create_qiskit_kraus(gate_params: list[np.ndarray]) -> Instruction:
    """create qiskit.quantum_info.Kraus from Braket Kraus operators and reorder axes"""
    for i, param in enumerate(gate_params):
        assert param.shape[0] == param.shape[1], "Kraus operators must be square matrices."
        gate_params[i] = _reverse_endianness(param)
    return qiskit_qi.Kraus(gate_params)


def _reverse_endianness(matrix: np.ndarray) -> np.ndarray:
    n_q = int(np.log2(matrix.shape[0]))
    # Convert multi-qubit Kraus from little to big endian notation
    return (
        np.transpose(
            matrix.reshape([2] * n_q * 2),
            list(range(n_q))[::-1] + list(range(n_q, 2 * n_q))[::-1],
        ).reshape((2**n_q, 2**n_q))
        if n_q > 1
        else matrix
    )


def _create_qiskit_gate(
    gate_name: str,
    gate_params: list[float | FreeParameterExpression],
    param_map: dict[str, Parameter],
) -> Instruction:
    gate_instance = _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(gate_name)
    if not gate_instance:
        raise TypeError(f'Braket gate "{gate_name}" not supported in Qiskit')
    new_gate_params = []
    for param_expression, value in zip(gate_instance.params, gate_params, strict=True):
        # extract the coefficient in the templated gate
        param = next(iter(param_expression.parameters)).sympify()
        coeff = float(param_expression.sympify().subs(param, 1))
        new_gate_params.append(
            sympy_to_qiskit(coeff * value.expression, param_map)
            if isinstance(value, FreeParameterExpression)
            else coeff * value
        )
    return gate_instance.__class__(*new_gate_params)


def convert_qiskit_to_braket_circuit(circuit: QuantumCircuit) -> Circuit:
    """Return a Braket quantum circuit from a Qiskit quantum circuit.

    Args:
        circuit (QuantumCircuit): Qiskit Quantum Circuit

    Returns:
        Circuit: Braket circuit
    """
    warnings.warn(
        "convert_qiskit_to_braket_circuit() is deprecated and "
        "will be removed in a future release. "
        "Use to_braket() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_braket(circuit)


def convert_qiskit_to_braket_circuits(
    circuits: list[QuantumCircuit],
) -> Iterable[Circuit]:
    """Converts all Qiskit circuits to Braket circuits.

    Args:
        circuits (List(QuantumCircuit)): Qiskit quantum circuit

    Returns:
        Iterable[Circuit]: Braket circuit
    """
    warnings.warn(
        "convert_qiskit_to_braket_circuits() is deprecated and "
        "will be removed in a future release. "
        "Use to_braket() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    for circuit in circuits:
        yield to_braket(circuit)
