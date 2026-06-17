"""OpenQASM 3 to Qiskit circuit context.

This module provides _QiskitProgramContext which interprets OpenQASM 3 programs
into Qiskit circuits, handling conditions, loops, verbatim markers, and
mid-circuit measurement.
"""

from collections.abc import Iterator, Sequence
from math import prod
from numbers import Number
from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit import (
    BoxOp,
    CircuitInstruction,
    Clbit,
    ForLoopOp,
    Gate,
    IfElseOp,
    Parameter,
    ParameterExpression,
    Qubit,
    WhileLoopOp,
)
from sympy import Add, Expr, Mul, Pow, Symbol, acos, asin, atan, cos, exp, log, sin, tan

from braket.default_simulator.openqasm._helpers.arrays import convert_range_def_to_range
from braket.default_simulator.openqasm._helpers.casting import cast_to
from braket.default_simulator.openqasm._helpers.functions import (
    evaluate_binary_expression,
    evaluate_unary_expression,
)
from braket.default_simulator.openqasm.interpreter import VerbatimBoxDelimiter
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
from qiskit_braket_provider.providers.gate_mappings import (
    _BRAKET_GATE_NAME_TO_QISKIT_GATE,
    _BRAKET_VERBATIM_BOX_NAME,
)

_SYMPY_FUNCTION_TO_QISKIT_METHOD = {
    sin: "sin",
    cos: "cos",
    tan: "tan",
    asin: "arcsin",
    acos: "arccos",
    atan: "arctan",
    exp: "exp",
    log: "log",
}


def _qiskit_numeric_power(exp: Expr) -> int | float:
    if not getattr(exp, "is_number", False) or not getattr(exp, "is_real", False):
        raise TypeError(f"unrecognized parameter type in conversion: {type(exp)}")
    return int(exp) if getattr(exp, "is_integer", False) else float(exp)


def _sympy_to_qiskit(
    expr: Expr, param_map: dict[str, Parameter]
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
            return _sympy_to_qiskit(base, param_map) ** _qiskit_numeric_power(exp)
        case obj if getattr(obj, "is_number", False) and getattr(obj, "is_real", False):
            return float(obj)
        case obj if obj.func in _SYMPY_FUNCTION_TO_QISKIT_METHOD and len(obj.args) == 1:
            method_name = _SYMPY_FUNCTION_TO_QISKIT_METHOD[obj.func]
            return getattr(_sympy_to_qiskit(obj.args[0], param_map), method_name)()
    raise TypeError(f"unrecognized parameter type in conversion: {type(expr)}")


class QiskitProgramContext(AbstractProgramContext):
    """Program context for converting OpenQASM 3 programs to Qiskit circuits.

    This context extends AbstractProgramContext to build Qiskit QuantumCircuits from
    OpenQASM 3 programs. It supports Braket verbatim pragmas, which are converted to
    Qiskit BoxOp operations to preserve gate sequences that should not be optimized.

    Verbatim boxes are represented using Qiskit's native BoxOp construct, which treats
    a block of operations atomically. All verbatim boxes in a circuit use the same
    configurable label name.
    """

    num_qubits: int

    def __init__(self, verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME) -> None:
        """Initialize the Qiskit program context.

        Args:
            verbatim_box_name: Name to use for BoxOp labels when converting verbatim boxes.
                Default: "verbatim"
        """
        super().__init__()
        self._circuit_stack: list[QuantumCircuit] = [QuantumCircuit()]
        self._param_map: dict[str, Parameter] = {}
        self._in_verbatim_box = False
        self._verbatim_circuit: QuantumCircuit | None = None
        self._verbatim_box_name = verbatim_box_name
        self._clbit_offset: dict[str, int] = {}

    @property
    def _active_circuit(self) -> QuantumCircuit:
        """The circuit that instructions should be added to (top of stack)."""
        return self._circuit_stack[-1]

    @property
    def circuit(self) -> QuantumCircuit:
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
        value: Any = None,  # noqa: ANN401
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

    def add_phase_instruction(self, target: int | list[int], phase_value: float) -> None:  # noqa: ARG002
        self._active_circuit.global_phase += phase_value

    def add_gate_instruction(
        self,
        gate_name: str,
        target: tuple[int, ...],
        params: list[Any] | None,
        ctrl_modifiers: list[int],
        power: int,
    ) -> None:
        gate: Gate = _BRAKET_GATE_NAME_TO_QISKIT_GATE[gate_name].copy()
        params = (
            [float(param) if isinstance(param, Number) else param for param in params]  # type: ignore[arg-type]
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
        classical_targets: Sequence[int] | None = None,
        *,
        classical_destination: Identifier | IndexedIdentifier | None = None,  # noqa: ARG002
    ) -> None:
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

    def is_mcm_dependent(self, expression: Expression) -> bool:
        """Check if expression depends on a mid-circuit measurement result.

        Delegates to the base class for identifier-based checks using
        _mcm_dependent_scopes, but always returns True for RangeDefinition
        and DiscreteSet to force for-loops through evaluate_for_range
        (producing ForLoopOp).
        """
        if isinstance(expression, (RangeDefinition, DiscreteSet)):
            return True
        return super().is_mcm_dependent(expression)

    def iter_classical_scopes(self, expression: Expression):  # noqa: ARG002
        """Yield once since Qiskit circuit building doesn't do per-path branching."""
        yield

    def evaluate_condition(self, condition: Expression) -> Iterator[bool]:
        """Evaluate a branching condition using a circuit stack.

        Yields True (visit if-block) then False (visit else-block).
        Each yield pushes a new circuit onto the stack; after the interpreter
        visits the block, the circuit is popped and used to build an IfElseOp.
        """
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
        else:
            raise NotImplementedError(
                f"Unsupported condition type '{type(condition).__name__}' in branching condition. "
                f"Only Identifier, IndexExpression, and BinaryExpression (==) are supported."
            )

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
    ) -> Iterator[None]:
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

    def evaluate_while_condition(self, condition: Expression) -> Iterator[bool]:
        """Evaluate a while-loop condition, capturing the body into a WhileLoopOp."""
        # MCM path: resolve condition and capture body into WhileLoopOp
        main = self._active_circuit
        if isinstance(condition, (Identifier, IndexExpression)):
            resolved_condition = self._resolve_condition_from_identifier(condition)
        elif isinstance(condition, BinaryExpression):
            if condition.op != BinaryOperator["=="]:
                raise NotImplementedError(
                    f"Unsupported operator '{condition.op.name}' in while-loop condition. "
                    f"Only '==' is supported for mid-circuit measurement while loops."
                )
            resolved_condition = self._resolve_condition(condition)
        else:
            raise NotImplementedError(
                f"Unsupported condition type '{type(condition).__name__}' in while-loop condition. "
                f"Only Identifier, IndexExpression, and BinaryExpression (==) are supported."
            )

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

    def _evaluate_expression(self, expression: Expression | list[Expression]) -> Any:  # noqa: ANN401
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
