"""Util function for provider.

This module provides utilities for converting between Braket and Qiskit quantum circuits,
including support for Braket verbatim pragmas. Verbatim boxes in OpenQASM 3 programs are
converted to Qiskit BoxOp operations, which treat blocks of gates atomically to preserve
sequences that should not be optimized.
"""

import warnings
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, overload

import numpy as np
import qiskit.circuit.library as qiskit_gates
import qiskit.quantum_info as qiskit_qi
from qiskit import QuantumCircuit
from qiskit.circuit import (
    ControlledGate,
    Parameter,
    ParameterExpression,
    ParameterVectorElement,
)
from qiskit.circuit import Instruction as QiskitInstruction
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.transpiler import PassManager, Target
from qiskit_ionq import add_equivalences

from braket.circuits import Circuit, Instruction, measure
from braket.circuits import Observable as BraketObservable
from braket.circuits import observables as braket_observables
from braket.default_simulator.openqasm.interpreter import Interpreter
from braket.devices import Device
from braket.ir.openqasm import Program
from braket.parametric import FreeParameter, FreeParameterExpression, Parameterizable
from qiskit_braket_provider.providers.compilation import (
    _T,  # noqa: F401
    _CompilationContext,  # noqa: F401
    _default_target,  # noqa: F401
    _extract_verbatim_boxes,  # noqa: F401
    _get_circuits,  # noqa: F401
    _restore_verbatim_boxes,  # noqa: F401
    _Translatable,
    compile_circuits,
)
from qiskit_braket_provider.providers.gate_mappings import (
    _BRAKET_GATE_NAME_TO_QISKIT_GATE,
    _BRAKET_TO_QISKIT_NAMES,  # noqa: F401
    _BRAKET_VERBATIM_BOX_NAME,
    _CONTROLLED_GATES_BY_QUBIT_COUNT,  # noqa: F401
    _QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES,
    _QISKIT_GATE_NAME_TO_BRAKET_GATE,
)
from qiskit_braket_provider.providers.qasm_context import (
    QiskitProgramContext,
    _sympy_to_qiskit,
)
from qiskit_braket_provider.providers.target import (
    SubstitutedTarget,  # noqa: F401
    _get_controlled_gateset,  # noqa: F401
    aws_device_to_target,  # noqa: F401
    gateset_from_properties,  # noqa: F401
    local_simulator_to_target,  # noqa: F401
    native_angle_restrictions,  # noqa: F401
    native_gate_connectivity,  # noqa: F401
    native_gate_set,  # noqa: F401
)

add_equivalences()

_EPS = 1e-10  # global variable used to chop very small numbers to zero

_BRAKET_SUPPORTED_NOISES = [
    "kraus",
    "bitflip",
    "depolarizing",
    "amplitudedamping",
    "generalizedamplitudedamping",
    "phasedamping",
    "phaseflip",
    "paulichannel",
    "twoqubitdepolarizing",
    "twoqubitdephasing",
    # "twoqubitpaulichannel" no to_openqasm support yet
]

_PAULI_MAP = {
    "X": braket_observables.X,
    "Y": braket_observables.Y,
    "Z": braket_observables.Z,
}


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
    result = compile_circuits(
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


def _create_qiskit_kraus(gate_params: list[np.ndarray]) -> QiskitInstruction:
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
) -> QiskitInstruction:
    gate_instance = _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(gate_name)
    if not gate_instance:
        raise TypeError(f'Braket gate "{gate_name}" not supported in Qiskit')
    new_gate_params = []
    for param_expression, value in zip(gate_instance.params, gate_params, strict=True):
        # extract the coefficient in the templated gate
        param = next(iter(param_expression.parameters)).sympify()
        coeff = float(param_expression.sympify().subs(param, 1))
        new_gate_params.append(
            _sympy_to_qiskit(coeff * value.expression, param_map)
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
