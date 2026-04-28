"""Util function for provider.

This module provides utilities for converting between Braket and Qiskit quantum circuits,
including support for Braket verbatim pragmas. Verbatim boxes in OpenQASM 3 programs are
converted to Qiskit BoxOp operations, which treat blocks of gates atomically to preserve
sequences that should not be optimized.
"""

import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import inf, pi, prod
from numbers import Number
from typing import TypeVar

import numpy as np
import qiskit.circuit.library as qiskit_gates
import qiskit.quantum_info as qiskit_qi
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import (
    Barrier,
    BoxOp,
    CircuitInstruction,
    Clbit,
    ControlledGate,
    ForLoopOp,
    Gate,
    IfElseOp,
    Measure,
    Parameter,
    ParameterExpression,
    ParameterVectorElement,
    Qubit,
    WhileLoopOp,
)
from qiskit.circuit import Instruction as QiskitInstruction
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.transpiler import (
    InstructionProperties,
    PassManager,
    QubitProperties,
    Target,
    TransformationPass,
)
from qiskit_ionq import add_equivalences, ionq_gates
from sympy import Add, Expr, Mul, Pow, Symbol

from braket import experimental_capabilities as braket_expcaps
from braket.aws import AwsDevice, AwsDeviceType
from braket.circuits import Circuit, Instruction, measure
from braket.circuits import Observable as BraketObservable
from braket.circuits import gates as braket_gates
from braket.circuits import noises as braket_noises
from braket.circuits import observables as braket_observables
from braket.default_simulator.openqasm._helpers.arrays import convert_range_def_to_range
from braket.default_simulator.openqasm._helpers.casting import cast_to
from braket.default_simulator.openqasm._helpers.functions import (
    evaluate_binary_expression,
    evaluate_unary_expression,
)
from braket.default_simulator.openqasm.interpreter import Interpreter, VerbatimBoxDelimiter
from braket.default_simulator.openqasm.parser.openqasm_ast import (
    ArrayLiteral,
    BinaryExpression,
    BinaryOperator,
    BitType,
    BooleanLiteral,
    Cast,
    ClassicalType,
    DiscreteSet,
    Expression,
    FloatLiteral,
    Identifier,
    IndexedIdentifier,
    IndexExpression,
    IntegerLiteral,
    RangeDefinition,
    SymbolLiteral,
    UnaryExpression,
)
from braket.default_simulator.openqasm.program_context import (
    AbstractProgramContext,
)
from braket.device_schema import (
    DeviceActionType,
    DeviceCapabilities,
    OpenQASMDeviceActionProperties,
)
from braket.device_schema.ionq import IonqDeviceCapabilities
from braket.device_schema.rigetti import RigettiDeviceCapabilities, RigettiDeviceCapabilitiesV2
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.device_schema.standardized_gate_model_qpu_device_properties_v1 import (
    StandardizedGateModelQpuDeviceProperties as StandardizedPropertiesV1,
)
from braket.device_schema.standardized_gate_model_qpu_device_properties_v2 import (
    StandardizedGateModelQpuDeviceProperties as StandardizedPropertiesV2,
)
from braket.devices import Device, LocalSimulator
from braket.ir.openqasm import Program
from braket.ir.openqasm.modifiers import Control
from braket.parametric import FreeParameter, FreeParameterExpression, Parameterizable
from qiskit_braket_provider.exception import QiskitBraketException
from qiskit_braket_provider.providers import braket_instructions

add_equivalences()

_BRAKET_TO_QISKIT_NAMES = {
    "u": "u",
    "phaseshift": "p",
    "cnot": "cx",
    "x": "x",
    "y": "y",
    "z": "z",
    "t": "t",
    "ti": "tdg",
    "s": "s",
    "si": "sdg",
    "v": "sx",
    "vi": "sxdg",
    "swap": "swap",
    "iswap": "iswap",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "xx": "rxx",
    "yy": "ryy",
    "zz": "rzz",
    "i": "id",
    "h": "h",
    "cy": "cy",
    "cz": "cz",
    "ccnot": "ccx",
    "cswap": "cswap",
    "cphaseshift": "cp",
    "ecr": "ecr",
    "prx": "r",
    "gpi": "gpi",
    "gpi2": "gpi2",
    "ms": "ms",
    "gphase": "global_phase",
    "unitary": "unitary",
    "kraus": "kraus",
}

_CONTROLLED_GATES_BY_QUBIT_COUNT = {
    1: {
        "ch": "h",
        "cs": "s",
        "csdg": "sdg",
        "csx": "sx",
        "crx": "rx",
        "cry": "ry",
        "crz": "rz",
        "ccz": "cz",
    },
    3: {"c3sx": "sx"},
    inf: {"mcx": "cx"},
}

_ADDITIONAL_U_GATES = {"u1", "u2", "u3"}

_EPS = 1e-10  # global variable used to chop very small numbers to zero

_QISKIT_GATE_NAME_TO_BRAKET_GATE: dict[str, Callable] = {
    "u1": lambda lam: [braket_gates.U(0, 0, lam)],
    "u2": lambda phi, lam: [braket_gates.U(pi / 2, phi, lam)],
    "u3": lambda theta, phi, lam: [braket_gates.U(theta, phi, lam)],
    "u": lambda theta, phi, lam: [braket_gates.U(theta, phi, lam)],
    "p": lambda angle: [braket_gates.PhaseShift(angle)],
    "cp": lambda angle: [braket_gates.CPhaseShift(angle)],
    "cx": lambda: [braket_gates.CNot()],
    "x": lambda: [braket_gates.X()],
    "y": lambda: [braket_gates.Y()],
    "z": lambda: [braket_gates.Z()],
    "t": lambda: [braket_gates.T()],
    "tdg": lambda: [braket_gates.Ti()],
    "s": lambda: [braket_gates.S()],
    "sdg": lambda: [braket_gates.Si()],
    "sx": lambda: [braket_gates.V()],
    "sxdg": lambda: [braket_gates.Vi()],
    "swap": lambda: [braket_gates.Swap()],
    "rx": lambda angle: [braket_gates.Rx(angle)],
    "ry": lambda angle: [braket_gates.Ry(angle)],
    "rz": lambda angle: [braket_gates.Rz(angle)],
    "rzz": lambda angle: [braket_gates.ZZ(angle)],
    "id": lambda: [braket_gates.I()],
    "h": lambda: [braket_gates.H()],
    "cy": lambda: [braket_gates.CY()],
    "cz": lambda: [braket_gates.CZ()],
    "ccx": lambda: [braket_gates.CCNot()],
    "cswap": lambda: [braket_gates.CSwap()],
    "rxx": lambda angle: [braket_gates.XX(angle)],
    "ryy": lambda angle: [braket_gates.YY(angle)],
    "ecr": lambda: [braket_gates.ECR()],
    "iswap": lambda: [braket_gates.ISwap()],
    "r": lambda angle_1, angle_2: [braket_gates.PRx(angle_1, angle_2)],
    # IonQ gates
    "gpi": lambda turns: [braket_gates.GPi(2 * pi * turns)],
    "gpi2": lambda turns: [braket_gates.GPi2(2 * pi * turns)],
    "ms": lambda turns_1, turns_2, turns_3: [
        braket_gates.MS(2 * pi * turns_1, 2 * pi * turns_2, 2 * pi * turns_3)
    ],
    "zz": lambda angle: [braket_gates.ZZ(2 * pi * angle)],
    # Global phase
    "global_phase": lambda phase: [braket_gates.GPhase(phase)],
    "unitary": lambda operators: [braket_gates.Unitary(operators[0])],
    "kraus": lambda operators: [braket_noises.Kraus(operators)],
    "CCPRx": lambda angle_1, angle_2, feedback_key: [
        braket_expcaps.iqm.classical_control.CCPRx(angle_1, angle_2, feedback_key)
    ],
    "MeasureFF": lambda feedback_key: [
        braket_expcaps.iqm.classical_control.MeasureFF(feedback_key)
    ],
}

_QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES: dict[str, Callable] = {
    controlled_gate: _QISKIT_GATE_NAME_TO_BRAKET_GATE[base_gate]
    for gate_map in _CONTROLLED_GATES_BY_QUBIT_COUNT.values()
    for controlled_gate, base_gate in gate_map.items()
}

_STANDARD_GATE_NAME_MAPPING = get_standard_gate_name_mapping()

_BRAKET_GATE_NAME_TO_QISKIT_GATE: dict[str, QiskitInstruction | None] = {
    "u": qiskit_gates.UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
    "u1": qiskit_gates.U1Gate(Parameter("theta")),
    "u2": qiskit_gates.U2Gate(Parameter("theta"), Parameter("lam")),
    "u3": qiskit_gates.U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
    "h": qiskit_gates.HGate(),
    "ccnot": qiskit_gates.CCXGate(),
    "cnot": qiskit_gates.CXGate(),
    "cphaseshift": qiskit_gates.CPhaseGate(Parameter("theta")),
    "cswap": qiskit_gates.CSwapGate(),
    "cy": qiskit_gates.CYGate(),
    "cz": qiskit_gates.CZGate(),
    "i": qiskit_gates.IGate(),
    "phaseshift": qiskit_gates.PhaseGate(Parameter("theta")),
    "rx": qiskit_gates.RXGate(Parameter("theta")),
    "ry": qiskit_gates.RYGate(Parameter("theta")),
    "rz": qiskit_gates.RZGate(Parameter("phi")),
    "s": qiskit_gates.SGate(),
    "si": qiskit_gates.SdgGate(),
    "swap": qiskit_gates.SwapGate(),
    "t": qiskit_gates.TGate(),
    "ti": qiskit_gates.TdgGate(),
    "v": qiskit_gates.SXGate(),
    "vi": qiskit_gates.SXdgGate(),
    "x": qiskit_gates.XGate(),
    "xx": qiskit_gates.RXXGate(Parameter("theta")),
    "y": qiskit_gates.YGate(),
    "yy": qiskit_gates.RYYGate(Parameter("theta")),
    "z": qiskit_gates.ZGate(),
    "zz": qiskit_gates.RZZGate(Parameter("theta")),
    "ecr": qiskit_gates.ECRGate(),
    "prx": qiskit_gates.RGate(Parameter("theta"), Parameter("phi")),
    "iswap": qiskit_gates.iSwapGate(),
    "gpi": ionq_gates.GPIGate(Parameter("phi") / (2 * pi)),
    "gpi2": ionq_gates.GPI2Gate(Parameter("phi") / (2 * pi)),
    "ms": ionq_gates.MSGate(
        Parameter("phi0") / (2 * pi),
        Parameter("phi1") / (2 * pi),
        Parameter("theta") / (2 * pi),
    ),
    "gphase": qiskit_gates.GlobalPhaseGate(Parameter("theta")),
    "measure": qiskit_gates.Measure(),
    "unitary": qiskit_gates.UnitaryGate,
    "kraus": qiskit_qi.Kraus,
    "cc_prx": braket_instructions.CCPRx(
        Parameter("angle_1"), Parameter("angle_2"), Parameter("feedback_key")
    ),
    "measure_ff": braket_instructions.MeasureFF(Parameter("feedback_key")),
}

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

_TRANSPILER_GATE_SUBSTITUTES: dict[tuple[str, tuple[float | str, ...]], Gate] = {
    ("rx", (pi,)): qiskit_gates.XGate(),
    ("rx", (-pi,)): qiskit_gates.XGate(),
    ("rx", (pi / 2,)): qiskit_gates.SXGate(),
    ("rx", (-pi / 2,)): qiskit_gates.SXdgGate(),
}

_PAULI_MAP = {
    "X": braket_observables.X,
    "Y": braket_observables.Y,
    "Z": braket_observables.Z,
}

_BRAKET_VERBATIM_BOX_NAME = "verbatim"

_Translatable = QuantumCircuit | Circuit | Program | str

_T = TypeVar("_T")


class _SubstitutedTarget(Target):
    def __new__(cls, *args, **kwargs):
        out = super().__new__(cls, *args, **kwargs)
        gate_substitutes = {}
        out._gate_substitutes = gate_substitutes
        out._pass_manager = PassManager([_SubstituteGates(gate_substitutes)])
        return out

    def _substitute(self, circuits: QuantumCircuit | Iterable[QuantumCircuit]):
        return self._pass_manager.run(circuits)


class _SubstituteGates(TransformationPass):
    def __init__(self, gate_substitutes: Mapping[str, Mapping[tuple[int, ...], QiskitInstruction]]):
        super().__init__()
        self._gate_substitutes = gate_substitutes

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if not self._gate_substitutes:
            return dag
        qubits = {q: i for i, q in enumerate(dag.qubits)}
        for node in dag.op_nodes():
            if (op_name := node.op.name) in self._gate_substitutes:
                dag.substitute_node(
                    node, self._gate_substitutes[op_name][tuple(qubits[q] for q in node.qargs)]
                )
        return dag


class _QiskitProgramContext(AbstractProgramContext):
    """Program context for converting OpenQASM 3 programs to Qiskit circuits.

    This context extends AbstractProgramContext to build Qiskit QuantumCircuits from
    OpenQASM 3 programs. It supports Braket verbatim pragmas, which are converted to
    Qiskit BoxOp operations to preserve gate sequences that should not be optimized.

    Verbatim boxes are represented using Qiskit's native BoxOp construct, which treats
    a block of operations atomically. All verbatim boxes in a circuit use the same
    configurable label name.
    """

    def __init__(self, verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME):
        """Initialize the Qiskit program context.

        Args:
            verbatim_box_name: Name to use for BoxOp labels when converting verbatim boxes.
                Default: "verbatim"
        """
        super().__init__()
        self._circuit_stack: list[QuantumCircuit] = [QuantumCircuit()]
        self._param_map = {}
        self._in_verbatim_box = False
        self._verbatim_circuit: QuantumCircuit | None = None
        self._verbatim_box_name = verbatim_box_name
        self._clbit_offset: dict[str, int] = {}
        self._measured_bits: set[str] = set()

    @property
    def _active_circuit(self) -> QuantumCircuit:
        """The circuit that instructions should be added to (top of stack)."""
        return self._circuit_stack[-1]

    @property
    def circuit(self):
        if self._in_verbatim_box:
            raise ValueError(
                "Unclosed verbatim box at end of program. "
                "Every verbatim box start marker must have a matching end marker."
            )
        return self._circuit_stack[0]

    def add_qubits(self, name: str, num_qubits: int | None = 1) -> None:
        super().add_qubits(name, num_qubits)
        self._active_circuit.add_register(num_qubits)

    def declare_variable(
        self,
        name: str,
        symbol_type: ClassicalType | type,
        value=None,
        const: bool = False,
    ) -> None:
        """Override to add classical bits to the Qiskit circuit when declared.

        When a classical bit array is declared (e.g., bit[2] c;), we need to add
        the corresponding classical bits to the Qiskit circuit.

        Note: This only adds classical bits when the size is known at declaration time.
        For function parameters with variable sizes (e.g., bit[n] where n is a parameter),
        the classical bits are not added since the size is not yet determined.
        """
        super().declare_variable(name, symbol_type, value, const)

        # If this is a bit type declaration, add classical bits to the circuit
        if isinstance(symbol_type, BitType):
            if symbol_type.size is not None:
                if isinstance(symbol_type.size, IntegerLiteral):
                    size = symbol_type.size.value
                else:
                    # Size is an Identifier or expression, can't determine size yet
                    # This happens for function parameters like bit[n] where n is a variable
                    return
            else:
                size = 1

            # this is used to deal with Qiskit's QuantumCircuit storing all classical bits in a flat list
            self._clbit_offset[name] = self._active_circuit.num_clbits
            self._active_circuit.add_bits([Clbit() for _ in range(size)])

    def is_builtin_gate(self, name: str) -> bool:
        return name in _BRAKET_GATE_NAME_TO_QISKIT_GATE

    def add_phase_instruction(self, target, phase_value):  # noqa: ARG002
        self._active_circuit.global_phase += phase_value

    def add_gate_instruction(
        self, gate_name: str, target: tuple[int, ...], params, ctrl_modifiers: list[int], power: int
    ):
        gate: Gate = _BRAKET_GATE_NAME_TO_QISKIT_GATE[gate_name].copy()
        params = (
            [float(param) if isinstance(param, Number) else param for param in params]
            if params is not None
            else []
        )
        if params:
            gate.params = params
        gate = gate.power(float(power)) if power != 1 else gate
        if ctrl_modifiers:
            gate = gate.control(
                len(ctrl_modifiers), ctrl_state=str("".join([str(i) for i in ctrl_modifiers]))
            )

        active = self._active_circuit
        # Ensure circuit has enough qubits for the target indices by adding missing qubits
        # This is needed when using physical qubits ($0, $1, etc.) where no qubit register is declared
        max_qubits = (max(target) + 1) if target else -1
        num_missing_qubits = max_qubits - active.num_qubits
        active.add_bits([Qubit() for _ in range(num_missing_qubits)])
        self.num_qubits = max(self.num_qubits, active.num_qubits)

        if self._in_verbatim_box:
            # Ensure verbatim circuit also has enough qubits by adding missing qubits
            num_missing_qubits = max_qubits - self._verbatim_circuit.num_qubits
            self._verbatim_circuit.add_bits([Qubit() for _ in range(num_missing_qubits)])
            self._verbatim_circuit.append(CircuitInstruction(gate, target))
        else:
            active.append(CircuitInstruction(gate, target))

    def handle_parameter_value(self, value: Number | Expr) -> Number | Parameter:
        return _sympy_to_qiskit(value, self._param_map) if isinstance(value, Expr) else value

    def add_measure(
        self,
        target: tuple[int],
        classical_targets: Iterable[int] | None = None,
        *,
        classical_destination: Identifier | IndexedIdentifier | None = None,
    ) -> None:
        if classical_destination is not None:
            name = (
                classical_destination.name
                if isinstance(classical_destination, IndexedIdentifier)
                else classical_destination
            )
            self._measured_bits.add(name.name)
        active = self._active_circuit
        # this is to cover the edge case where a user measures a qubit without assigning it to a classical register
        if active.num_clbits < len(target):
            num_missing_clbits = len(target) - active.num_clbits
            active.add_bits([Clbit() for _ in range(num_missing_clbits)])
        for idx, qubit in enumerate(target):
            index = classical_targets[idx] if classical_targets else idx
            active.measure(qubit, index)

    def add_verbatim_marker(self, marker: VerbatimBoxDelimiter) -> None:
        """Handle verbatim box start/end markers.

        When START_VERBATIM is received:
        - Create a new QuantumCircuit to collect verbatim gates
        - Set _in_verbatim_box flag to True

        When END_VERBATIM is received:
        - Wrap the collected gates in a BoxOp
        - Append the BoxOp to the main circuit
        - Reset verbatim state

        Args:
            marker: VerbatimBoxDelimiter indicating START_VERBATIM or END_VERBATIM

        Raises:
            ValueError: If nested verbatim boxes are encountered or if END_VERBATIM is called without START_VERBATIM
        """

        if marker == VerbatimBoxDelimiter.START_VERBATIM:
            if self._in_verbatim_box:
                raise ValueError("Nested verbatim boxes are not supported")

            self._verbatim_circuit = QuantumCircuit()
            self._in_verbatim_box = True

        elif marker == VerbatimBoxDelimiter.END_VERBATIM:
            if not self._in_verbatim_box:
                raise ValueError("Verbatim box end marker without matching start")

            box_op = BoxOp(self._verbatim_circuit, label=self._verbatim_box_name)

            active = self._active_circuit
            # Append BoxOp to active circuit with all qubits (convert indices to Qubit objects)
            qubit_objects = [active.qubits[i] for i in range(self._verbatim_circuit.num_qubits)]
            active.append(box_op, qubit_objects)

            self._in_verbatim_box = False
            self._verbatim_circuit = None

        else:
            raise ValueError("Verbatim box created using invalid marker")

    @property
    def supports_midcircuit_measurement(self) -> bool:
        return True

    def _references_measurement(self, condition) -> bool:
        """Check if condition references a variable that was measured into."""
        match condition:
            case Identifier(name=name):
                return name in self._measured_bits
            case IndexExpression(collection=Identifier(name=name)):
                return name in self._measured_bits
            case BinaryExpression(lhs=lhs, rhs=rhs):
                return self._references_measurement(lhs) or self._references_measurement(rhs)
            case _:
                return False

    def evaluate_condition(self, condition):
        """Evaluate a branching condition using a circuit stack.

        Yields True (visit if-block) then False (visit else-block).
        Each yield pushes a new circuit onto the stack; after the interpreter
        visits the block, the circuit is popped and used to build an IfElseOp.

        For static conditions (no measurement dependency), evaluates directly
        and yields only the taken branch.
        """
        if not self._references_measurement(condition):
            try:
                result = cast_to(BooleanLiteral, self._evaluate_expression(condition))
            except (TypeError, ValueError, AttributeError) as e:
                raise TypeError("Unsupported condition in branching statement") from e
            yield result.value
            return

        # MCM path: resolve condition to (Clbit, value)
        main = self._active_circuit
        if isinstance(condition, (Identifier, IndexExpression)):
            # Bare identifier like `if (c)` or `if (c[0])` — equivalent to `== 1`
            resolved_condition = self._resolve_condition_from_identifier(condition)
        elif isinstance(condition, BinaryExpression):
            if condition.op != BinaryOperator["=="]:
                raise NotImplementedError(
                    f"Unsupported operator '{condition.op.name}' in branching condition. "
                    f"Only '==' is supported for mid-circuit measurement branching."
                )
            resolved_condition = self._resolve_condition(condition)

        true_body = QuantumCircuit(main.num_qubits, main.num_clbits)
        self._circuit_stack.append(true_body)
        yield True
        self._circuit_stack.pop()

        # Push circuit for else-block (the interpreter always consumes this yield
        # even for if-only blocks; the empty circuit is discarded below)
        false_body = QuantumCircuit(main.num_qubits, main.num_clbits)
        self._circuit_stack.append(false_body)
        yield False
        self._circuit_stack.pop()

        actual_false = false_body if false_body.data else None

        if not true_body.data and not actual_false:
            raise ValueError(
                "Branching statement conditioned on a measurement has empty bodies. "
                "Both if and else branches contain no quantum operations."
            )

        # Sync main circuit dimensions if branch bodies grew
        max_qubits = max(true_body.num_qubits, false_body.num_qubits)
        max_clbits = max(true_body.num_clbits, false_body.num_clbits)
        if max_qubits > main.num_qubits:
            main.add_bits([Qubit() for _ in range(max_qubits - main.num_qubits)])
        if max_clbits > main.num_clbits:
            main.add_bits([Clbit() for _ in range(max_clbits - main.num_clbits)])

        if_else_op = IfElseOp(resolved_condition, true_body, actual_false)
        qubits = list(range(max_qubits))
        clbits = list(range(max_clbits))
        main.append(if_else_op, qubits, clbits)

    def evaluate_for_range(
        self, set_declaration: Expression, loop_var: str, loop_type: ClassicalType
    ):
        """Capture the for-loop body into a ForLoopOp.

        Yields once to capture the body. If the body contains quantum operations,
        wraps it in a ForLoopOp. If purely classical (empty body circuit),
        falls back to static unrolling for remaining iterations.
        """
        index = self._evaluate_expression(set_declaration)
        if isinstance(index, RangeDefinition):
            index_values = [IntegerLiteral(x) for x in convert_range_def_to_range(index)]
        else:
            index_values = index.values

        if not index_values:
            return

        main = self._active_circuit
        # First pass: capture with concrete value to detect classical vs quantum body
        # Snapshot outer scope variables to detect classical side effects
        outer_vars = dict(self.variable_table.current_scope)
        probe = QuantumCircuit(main.num_qubits, main.num_clbits)
        self._circuit_stack.append(probe)
        with self.enter_scope():
            self.declare_variable(loop_var, loop_type, index_values[0])
            yield
        self._circuit_stack.pop()

        if not probe.data:
            # Purely classical loop body — statically unroll remaining iterations
            for i in index_values[1:]:
                with self.enter_scope():
                    self.declare_variable(loop_var, loop_type, i)
                    yield
            return

        # Detect classical side effects: if outer scope variables changed during
        # the probe, the body mutates classical state that ForLoopOp won't capture
        modified_vars = [
            name
            for name, val in self.variable_table.current_scope.items()
            if name in outer_vars and outer_vars[name] != val
        ]
        if modified_vars:
            raise ValueError(
                f"For loop body modifies classical variable(s) {modified_vars} "
                f"which cannot be captured by ForLoopOp. "
                f"Only quantum operations are preserved across loop iterations."
            )

        # Second pass: re-capture with symbolic loop variable for correct ForLoopOp binding
        body = QuantumCircuit(main.num_qubits, main.num_clbits)
        self._circuit_stack.append(body)
        with self.enter_scope():
            self.declare_variable(loop_var, loop_type, SymbolLiteral(Symbol(loop_var)))
            yield
        self._circuit_stack.pop()

        indexset = tuple(iv.value for iv in index_values)
        loop_param = self._param_map.get(loop_var) or Parameter(loop_var)
        for_op = ForLoopOp(indexset, loop_param, body)
        max_qubits = max(body.num_qubits, main.num_qubits)
        max_clbits = max(body.num_clbits, main.num_clbits)
        if max_qubits > main.num_qubits:
            main.add_bits([Qubit() for _ in range(max_qubits - main.num_qubits)])
        if max_clbits > main.num_clbits:
            main.add_bits([Clbit() for _ in range(max_clbits - main.num_clbits)])
        main.append(for_op, list(range(max_qubits)), list(range(max_clbits)))

    def handle_loop_break(self):
        """Reject break statements since Qiskit's ForLoopOp/WhileLoopOp do not support them."""
        raise NotImplementedError("break statements are not supported in loops.")

    def handle_loop_continue(self):
        """Reject continue statements since Qiskit's ForLoopOp/WhileLoopOp do not support them."""
        raise NotImplementedError("continue statements are not supported in loops.")

    def evaluate_while_condition(self, condition: Expression):
        """Evaluate a while-loop condition, yielding True per iteration.

        For static conditions (no measurement dependency), evaluates directly.
        For MCM-dependent conditions, captures the body into a WhileLoopOp.
        """
        if not self._references_measurement(condition):
            try:
                while cast_to(BooleanLiteral, self._evaluate_expression(condition)).value:
                    yield True
            except (TypeError, ValueError, AttributeError) as e:
                raise TypeError("Unsupported condition in while loop") from e
            else:
                return

        # MCM path: resolve condition and capture body into WhileLoopOp
        main = self._active_circuit
        if isinstance(condition, (Identifier, IndexExpression)):
            resolved_condition = self._resolve_condition_from_identifier(condition)
        else:
            if condition.op != BinaryOperator["=="]:
                raise NotImplementedError(
                    f"Unsupported operator '{condition.op.name}' in while-loop condition. "
                    f"Only '==' is supported for mid-circuit measurement while loops."
                )
            resolved_condition = self._resolve_condition(condition)

        body = QuantumCircuit(main.num_qubits, main.num_clbits)
        self._circuit_stack.append(body)
        yield True
        self._circuit_stack.pop()

        if not body.data:
            raise ValueError(
                "While loop conditioned on a measurement has an empty body. "
                "This would result in an infinite loop."
            )

        max_qubits = max(body.num_qubits, main.num_qubits)
        max_clbits = max(body.num_clbits, main.num_clbits)
        if max_qubits > main.num_qubits:
            main.add_bits([Qubit() for _ in range(max_qubits - main.num_qubits)])
        if max_clbits > main.num_clbits:
            main.add_bits([Clbit() for _ in range(max_clbits - main.num_clbits)])

        while_op = WhileLoopOp(resolved_condition, body)
        main.append(while_op, list(range(max_qubits)), list(range(max_clbits)))

    def _evaluate_expression(self, expression):
        """Lightweight expression evaluator for loop conditions and ranges."""
        match expression:
            case (
                BooleanLiteral()
                | IntegerLiteral()
                | FloatLiteral()
                | ArrayLiteral()
                | SymbolLiteral()
            ):
                return expression
            case Identifier():
                return self.get_value_by_identifier(expression)
            case BinaryExpression(lhs=lhs, rhs=rhs, op=op):
                return evaluate_binary_expression(
                    self._evaluate_expression(lhs),
                    self._evaluate_expression(rhs),
                    op,
                )
            case UnaryExpression(expression=inner, op=op):
                return evaluate_unary_expression(self._evaluate_expression(inner), op)
            case Cast(type=cast_type, argument=argument):
                return cast_to(cast_type, self._evaluate_expression(argument))
            case RangeDefinition(start=start, end=end, step=step):
                return RangeDefinition(
                    self._evaluate_expression(start) if start else None,
                    self._evaluate_expression(end),
                    self._evaluate_expression(step) if step else None,
                )
            case DiscreteSet(values=values):
                return DiscreteSet(values=[self._evaluate_expression(v) for v in values])
            case list():
                return [self._evaluate_expression(item) for item in expression]
            case _:
                raise TypeError(f"Cannot evaluate expression of type {type(expression).__name__}")

    def _resolve_condition(self, condition: BinaryExpression) -> tuple[Clbit, int]:
        """Convert an OpenQASM condition AST node to a Qiskit (Clbit, int) condition."""
        if isinstance(condition.lhs, (Identifier, IndexExpression)):
            clbit_index = self._resolve_clbit_index(condition.lhs)
            value = condition.rhs.value
        else:
            clbit_index = self._resolve_clbit_index(condition.rhs)
            value = condition.lhs.value
        return (self.circuit.clbits[clbit_index], int(value))

    def _resolve_condition_from_identifier(
        self, condition: Identifier | IndexExpression
    ) -> tuple[Clbit, int]:
        """Convert a bare identifier condition (e.g., `c` or `c[0]`) to (Clbit, 1)."""
        clbit_index = self._resolve_clbit_index(condition)
        return (self.circuit.clbits[clbit_index], 1)

    def _resolve_clbit_index(self, node: Identifier | IndexExpression) -> int:
        """Resolve an identifier or indexed identifier to a classical bit index."""
        if isinstance(node, IndexExpression):
            name = node.collection.name
            index = node.index[0].value
        elif isinstance(node, Identifier):
            name = node.name
            var_type = self.get_type(name)
            if isinstance(var_type, BitType) and var_type.size is not None:
                size = var_type.size.value if isinstance(var_type.size, IntegerLiteral) else None
                if size is not None and size > 1:
                    raise TypeError(
                        f"Multi-bit register '{name}' (bit[{size}]) cannot be used as a "
                        f"single-bit condition. Use an indexed reference like '{name}[0]'."
                    )
            index = 0
        else:
            raise TypeError(f"Unsupported condition operand type: {type(node)}")

        return self._clbit_offset[name] + index


def native_gate_connectivity(properties: DeviceCapabilities) -> list[list[int]] | None:
    """Returns the connectivity natively supported by a Braket device from its properties

    Args:
        properties (DeviceCapabilities): The device properties of the Braket device.

    Returns:
        list[list[int]] | None: A list of connected qubit pairs or ``None``
        if the device is fully connected.
    """
    device_connectivity = properties.paradigm.connectivity
    return (
        [
            [int(x), int(y)]
            for x, neighborhood in device_connectivity.connectivityGraph.items()
            for y in neighborhood
        ]
        if not device_connectivity.fullyConnected
        else None
    )


def native_gate_set(properties: DeviceCapabilities) -> set[str]:
    """Returns the gate set natively supported by a Braket device from its properties

    Args:
        properties (DeviceCapabilities): The device properties of the Braket device.

    Returns:
        set[str]: The names of Qiskit gates natively supported by the Braket device.
    """
    native_list = properties.paradigm.nativeGateSet
    return {
        _BRAKET_TO_QISKIT_NAMES[op.lower()]
        for op in native_list
        if op.lower() in _BRAKET_TO_QISKIT_NAMES
    }


def native_angle_restrictions(
    properties: DeviceCapabilities,
) -> dict[str, dict[int, set[float] | tuple[float, float]]]:
    """Returns angle restrictions for gates natively supported by a Braket device.

    The returned mapping specifies, for each gate name, constraints on the
    gate parameters indexed by their position. The constraint can either be a
    set of allowed angles or a tuple representing an inclusive ``(min, max)``
    range. Angle units are in radians.

    Args:
        properties (DeviceCapabilities): The device properties of the Braket device.

    Returns:
        dict[str, dict[int, set[float] | tuple[float, float]]]: Mapping
        of gate names to parameter index restrictions.
    """

    if isinstance(properties, (RigettiDeviceCapabilities, RigettiDeviceCapabilitiesV2)):
        return {"rx": {0: {pi, -pi, pi / 2, -pi / 2}}}
    if isinstance(properties, IonqDeviceCapabilities):
        return {"ms": {2: (0.0, 0.25)}}
    return {}


def gateset_from_properties(properties: OpenQASMDeviceActionProperties) -> set[str]:
    """Returns the gateset supported by a Braket device with the given properties

    Args:
        properties (OpenQASMDeviceActionProperties): The action properties of the Braket device.

    Returns:
        set[str]: The names of the gates supported by the device
    """
    gateset = {
        _BRAKET_TO_QISKIT_NAMES[op.lower()]
        for op in properties.supportedOperations
        if op.lower() in _BRAKET_TO_QISKIT_NAMES
    }
    if "u" in gateset:
        gateset.update(_ADDITIONAL_U_GATES)
    max_control = 0
    for modifier in properties.supportedModifiers:
        if isinstance(modifier, Control):
            max_control = modifier.max_qubits
            break
    return gateset.union(_get_controlled_gateset(gateset, max_control))


def _get_controlled_gateset(base_gateset: set[str], max_qubits: int | None = None) -> set[str]:
    """Returns the Qiskit gates expressible as controlled versions of existing Braket gates

    This set can be filtered by the maximum number of control qubits.

    Args:
        base_gateset (set[str]): The base (without control modifiers) gates supported
        max_qubits (int | None): The maximum number of control qubits that can be used to express
            the Qiskit gate as a controlled Braket gate. If ``None``, then there is no limit to the
            number of control qubits. Default: ``None``.

    Returns:
        set[str]: The names of the controlled gates.
    """
    max_control = max_qubits if max_qubits is not None else inf
    return {
        controlled_gate
        for control_count, gate_map in _CONTROLLED_GATES_BY_QUBIT_COUNT.items()
        for controlled_gate, base_gate in gate_map.items()
        if control_count <= max_control and base_gate in base_gateset
    }


# TODO: move target construction to a dedicated file; AwsDevice target construction is getting big
def local_simulator_to_target(simulator: LocalSimulator) -> Target:
    """Converts properties of a Braket LocalSimulator into a Qiskit Target object.

    Args:
        simulator (LocalSimulator): Amazon Braket ``LocalSimulator``

    Returns:
        Target: Target for Qiskit backend
    """
    return _simulator_target(
        simulator, f"Target for Amazon Braket local simulator: {simulator.name}"
    )


def aws_device_to_target(device: AwsDevice) -> Target:
    """Converts properties of Braket AwsDevice into a Qiskit Target object.

    Args:
        device (AwsDevice): Amazon Braket ``AwsDevice``

    Returns:
        Target: Target for Qiskit backend
    """
    match device.type:
        case AwsDeviceType.QPU:
            return _qpu_target(device, f"Target for Amazon Braket QPU: {device.name}")
        case AwsDeviceType.SIMULATOR:
            return _simulator_target(device, f"Target for Amazon Braket simulator: {device.name}")
    raise QiskitBraketException(
        "Cannot convert to target. "
        f"{device.properties.__class__} device capabilities are not supported."
    )


def _simulator_target(device: Device, description: str):
    properties: GateModelSimulatorDeviceCapabilities = device.properties
    target = Target(description=description, num_qubits=properties.paradigm.qubitCount)
    action = properties.action.get(DeviceActionType.OPENQASM) or properties.action.get(
        DeviceActionType.JAQCD
    )
    for operation in action.supportedOperations:
        instruction = _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(operation.lower())
        if instruction:
            target.add_instruction(instruction, name=_BRAKET_TO_QISKIT_NAMES[operation.lower()])
    if isinstance(action, OpenQASMDeviceActionProperties):
        max_control = 0
        for modifier in action.supportedModifiers:
            if isinstance(modifier, Control):
                max_control = modifier.max_qubits
                break
        for gate in _get_controlled_gateset(target.keys(), max_control):
            if gate in _STANDARD_GATE_NAME_MAPPING:
                target.add_instruction(_STANDARD_GATE_NAME_MAPPING[gate])
    target.add_instruction(Measure())
    target.add_instruction(Barrier(1))
    return target


def _qpu_target(device: AwsDevice, description: str):
    properties: DeviceCapabilities = device.properties
    topology = device.topology_graph
    standardized = properties.standardized
    indices = {q: i for i, q in enumerate(sorted(topology.nodes))}

    qubit_properties = []
    default_instruction_props = InstructionProperties(error=0)
    instruction_props_measurement = {}
    instruction_props_1q = {}
    instruction_props_2q = {}
    # TODO: Support V3 standardized properties
    if isinstance(standardized, (StandardizedPropertiesV1, StandardizedPropertiesV2)):
        props_1q = standardized.oneQubitProperties
        for q in sorted(int(q) for q in props_1q):
            if q not in indices:
                warnings.warn(
                    f"Qubit {q} found in device properties but not in topology. "
                    f"Skipping qubit {q} and its associated properties.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            props = props_1q[str(q)]
            key = (indices[q],)
            for fidelity in props.oneQubitFidelity:
                match fidelity.fidelityType.name.lower():
                    case "readout":
                        instruction_props_measurement[key] = InstructionProperties(
                            # Use highest known error rate
                            error=max(
                                1 - fidelity.fidelity,
                                instruction_props_measurement.get(
                                    key, default_instruction_props
                                ).error,
                            )
                        )
                    case name if "readout_error" not in name:
                        instruction_props_1q[key] = InstructionProperties(
                            error=max(
                                1 - fidelity.fidelity,
                                instruction_props_1q.get(key, default_instruction_props).error,
                            )
                        )
            qubit_properties.append(QubitProperties(t1=props.T1.value, t2=props.T2.value))
        instruction_props_2q.update(
            _build_instruction_props_2q(standardized, indices, default_instruction_props)
        )

    default_props_1q = {(i,): None for i in indices.values()}
    default_props_2q = {(indices[u], indices[v]): None for u, v in topology.edges}
    if not instruction_props_measurement:
        instruction_props_measurement.update(default_props_1q)
    if not instruction_props_1q:
        instruction_props_1q.update(default_props_1q)

    parameter_restrictions = _get_parameter_restrictions(device, indices)
    target = _SubstitutedTarget(
        description=description,
        num_qubits=len(qubit_properties or indices),
        qubit_properties=qubit_properties or None,
    )
    if parameter_restrictions:
        _add_instructions_parameter_restrictions(
            target,
            parameter_restrictions,
            instruction_props_1q,
            instruction_props_2q,
            default_props_2q,
        )
    else:
        _add_instructions_no_parameter_restrictions(
            target,
            properties.paradigm.nativeGateSet,
            instruction_props_1q,
            instruction_props_2q,
            default_props_2q,
        )

    if "barrier" in properties.paradigm.nativeGateSet:
        target.add_instruction(Barrier(1))

    # Add measurement if not already added
    if "measure" not in target:
        target.add_instruction(Measure(), instruction_props_measurement)
    return target


def _build_instruction_props_2q(
    standardized: StandardizedPropertiesV1 | StandardizedPropertiesV2,
    indices: Mapping[int, int],
    default_properties: InstructionProperties,
) -> dict[str, dict[tuple[int, int], InstructionProperties]]:
    instruction_props_2q = defaultdict(dict)
    for k, props in standardized.twoQubitProperties.items():
        qubits = [int(q) for q in k.split("-")]
        # Check if all qubits in the edge exist in topology
        if not all(q in indices for q in qubits):
            missing_qubits = [q for q in qubits if q not in indices]
            warnings.warn(
                f"Edge {k} contains qubits {missing_qubits} not found in topology. "
                f"Skipping edge {k} and its associated properties.",
                UserWarning,
                stacklevel=2,
            )
            continue

        for fidelity in props.twoQubitGateFidelity:
            if gate_name := _BRAKET_TO_QISKIT_NAMES.get(fidelity.gateName.lower()):
                edge = tuple(indices[q] for q in qubits)
                instruction_props_2q[gate_name][edge] = InstructionProperties(
                    error=max(
                        1 - fidelity.fidelity,
                        instruction_props_2q[gate_name].get(edge, default_properties).error,
                    )
                )
    # Standardized 2q gate props assume bidirectionality
    for edge_props in instruction_props_2q.values():
        edge_props.update({
            tuple(reversed(edge)): instruction_props
            for edge, instruction_props in edge_props.items()
        })
    return instruction_props_2q


def _get_parameter_restrictions(
    device: AwsDevice, qubit_indices: Mapping[int, int]
) -> dict[str, dict[tuple[float | str, ...], set[tuple[int, ...]]]]:
    cal = device.gate_calibrations
    parameter_restrictions = defaultdict(lambda: defaultdict(set))
    for gate, target in cal.pulse_sequences if cal else {}:
        gate_name = gate.name.lower()
        qubits = tuple(qubit_indices[q] for q in target)
        if isinstance(gate, Parameterizable):
            param_key = tuple(
                param.name if isinstance(param, FreeParameter) else param
                for param in gate.parameters
            )
            parameter_restrictions[gate_name][tuple(param_key)].add(qubits)
        else:
            parameter_restrictions[gate_name][()].add(qubits)
    return parameter_restrictions


def _add_instructions_parameter_restrictions(
    target: _SubstitutedTarget,
    parameter_restrictions: Mapping[str, Mapping[tuple[float | str, ...], set[tuple[int, ...]]]],
    instruction_props_1q: Mapping[tuple[int], InstructionProperties],
    instruction_props_2q: Mapping[str, Mapping[tuple[int, int], InstructionProperties]],
    default_props_2q: Mapping[tuple[int, int], InstructionProperties | None],
) -> None:
    for braket_name, restrictions in parameter_restrictions.items():
        if instruction := _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(braket_name):
            gate_name = instruction.name
            match num_qubits := instruction.num_qubits:
                case 1:
                    _add_single_instruction_parameter_restriction(
                        target,
                        instruction,
                        braket_name,
                        restrictions,
                        instruction_props_1q,
                    )
                case 2:
                    _add_single_instruction_parameter_restriction(
                        target,
                        instruction,
                        braket_name,
                        restrictions,
                        instruction_props_2q.get(gate_name, default_props_2q),
                    )
                case _:
                    warnings.warn(
                        f"Instruction {gate_name} has {num_qubits} qubits "
                        "and cannot be added to target",
                        stacklevel=2,
                    )


def _add_single_instruction_parameter_restriction(
    target: _SubstitutedTarget,
    instruction: QiskitInstruction,
    braket_name: str,
    restrictions: Mapping[tuple[float | str, ...], set[tuple[int, ...]]],
    gate_properties: Mapping[tuple[int, ...], InstructionProperties],
) -> None:
    for restriction, qubits in restrictions.items():
        props = {q: props for q, props in gate_properties.items() if q in qubits}
        instruction_copy = instruction.copy()
        if restriction:
            instruction_copy.params = [
                Parameter(param) if isinstance(param, str) else param for param in restriction
            ]
        if substitute := _TRANSPILER_GATE_SUBSTITUTES.get((braket_name, restriction)):
            substitute_name = substitute.name
            if instruction_target := target.get(substitute_name):
                # Nothing to do with Hamiltonian mechanics; variable names are coincidental :)
                for q, p in props.items():
                    if not (current := instruction_target.get(q)) or current.error > p.error:
                        instruction_target[q] = p
                        target._gate_substitutes[substitute_name][q] = instruction_copy
            else:
                target.add_instruction(substitute, props)
                target._gate_substitutes[substitute_name] = dict.fromkeys(props, instruction_copy)
        else:
            target.add_instruction(instruction_copy, props)


def _add_instructions_no_parameter_restrictions(
    target: _SubstitutedTarget,
    native_gateset: set[str],
    instruction_props_1q: Mapping[tuple[int], InstructionProperties],
    instruction_props_2q: Mapping[str, Mapping[tuple[int, int], InstructionProperties]],
    default_props_2q: Mapping[tuple[int, int], InstructionProperties | None],
) -> None:
    for operation in native_gateset:
        if instruction := _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(operation.lower()):
            gate_name = instruction.name
            match num_qubits := instruction.num_qubits:
                case 1:
                    target.add_instruction(instruction, instruction_props_1q)
                case 2:
                    target.add_instruction(
                        instruction, instruction_props_2q.get(gate_name, default_props_2q)
                    )
                case _:
                    warnings.warn(
                        f"Instruction {gate_name} has {num_qubits} qubits "
                        "and cannot be added to target",
                        stacklevel=2,
                    )


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
    reconstructed_circuit = QuantumCircuit(
        transpiled_circuit.num_qubits, transpiled_circuit.num_clbits
    )
    reconstructed_circuit.global_phase = transpiled_circuit.global_phase

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
    basis_gates: Sequence[str] | None
    angle_restrictions: Mapping[str, Mapping[int, set[float] | tuple[float, float]]] | None
    pass_manager: PassManager | None


def _compile(
    circuits: _Translatable | Iterable[_Translatable] = None,
    *args,
    qubit_labels: Sequence[int] | None = None,
    target: Target | None = None,
    verbatim: bool | None = None,
    basis_gates: Sequence[str] | None = None,
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
    if isinstance(target, _SubstitutedTarget):
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


def to_braket(
    circuits: _Translatable | Iterable[_Translatable] = None,
    *args,
    qubit_labels: Sequence[int] | None = None,
    target: Target | None = None,
    verbatim: bool | None = None,
    basis_gates: Sequence[str] | None = None,
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
        basis_gates (Sequence[str] | None): The gateset to transpile to. Can only be provided
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
):
    if not (circuits or circuit):
        raise ValueError("Must specify circuits to transpile")
    if circuit:
        if circuits:
            raise ValueError("Cannot specify both circuits and circuit")
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
    basis_gates: Sequence[str] | None,
    coupling_map: list[list[int]] | None,
    connectivity: list[list[int]] | None,
    pass_manager: PassManager | None,
    braket_device: Device | None,
):
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
    circuit: _Translatable,
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
    measured_qubits = {}
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


def _create_free_parameters(operation):
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


def _validate_name_conflicts(parameters):
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
            Interpreter(_QiskitProgramContext(verbatim_box_name))
            .run(circuit.source, inputs=circuit.inputs)
            .circuit
        )
    if isinstance(circuit, str):
        return Interpreter(_QiskitProgramContext(verbatim_box_name)).run(circuit).circuit
    if not isinstance(circuit, Circuit):
        raise TypeError(f"Expected a Circuit, got {type(circuit)} instead.")

    num_measurements = sum(
        isinstance(instr.operator, measure.Measure) for instr in circuit.instructions
    )
    qiskit_circuit = QuantumCircuit(circuit.qubit_count, num_measurements)
    qubit_map = {int(qubit): index for index, qubit in enumerate(sorted(circuit.qubits))}
    parameter_map = {}
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


def _create_qiskit_unitary(matrix: np.ndarray):
    return qiskit_gates.UnitaryGate(_reverse_endianness(matrix))


def _create_qiskit_kraus(gate_params: list[np.ndarray]) -> Instruction:
    """create qiskit.quantum_info.Kraus from Braket Kraus operators and reorder axes"""
    for i, param in enumerate(gate_params):
        assert param.shape[0] == param.shape[1], "Kraus operators must be square matrices."
        gate_params[i] = _reverse_endianness(param)
    return qiskit_qi.Kraus(gate_params)


def _sympy_to_qiskit(
    expr: Expr, param_map: Mapping[str, Parameter]
) -> ParameterExpression | Parameter:
    """convert a sympy expression to qiskit Parameters recursively"""
    match expr:
        case Symbol(name=name):
            if name not in param_map:
                param_map[name] = Parameter(name)
            return param_map[name]
        case Add(args=args):
            return sum(_sympy_to_qiskit(arg, param_map) for arg in args)
        case Mul(args=args):
            return prod(_sympy_to_qiskit(arg, param_map) for arg in args)
        case Pow(base=base, exp=exp):
            return _sympy_to_qiskit(base, param_map) ** int(exp)
        case obj if getattr(obj, "is_real", False):
            return float(obj)
    raise TypeError(f"unrecognized parameter type in conversion: {type(expr)}")


def _reverse_endianness(matrix: np.ndarray):
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
    param_map: Mapping[str, Parameter],
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
