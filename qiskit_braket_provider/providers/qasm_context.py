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
from sympy import Add, Expr, Mul, Pow, Symbol

from braket.default_simulator.openqasm._helpers.arrays import convert_range_def_to_range
from braket.default_simulator.openqasm._helpers.casting import cast_to, get_identifier_name
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
from braket.ir.jaqcd.program_v1 import Results
from qiskit_braket_provider.providers.gate_mappings import (
    _BRAKET_GATE_NAME_TO_QISKIT_GATE,
    _BRAKET_VERBATIM_BOX_NAME,
    _SYMPY_FUNCTION_TO_QISKIT_METHOD,
)


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


class _QiskitProgramContext(AbstractProgramContext):
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
        self._verbatim_box_name = verbatim_box_name
        self._clbit_offset: dict[str, int] = {}
        self._result_types: list[dict[str, Any]] = []
        self._last_pragma_command: str = ""

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

    def parse_pragma(self, pragma_body: str):  # noqa: ANN202
        """Parse pragma and capture raw text for result type pragmas.

        Overrides AbstractProgramContext.parse_pragma() to store the raw pragma
        command string before parsing. This allows us to reconstruct the full
        pragma text for re-emission in the compiled output.

        Args:
            pragma_body: The body of the pragma statement (e.g., "braket result expectation z(q[0])").
        """
        self._last_pragma_command = pragma_body
        return super().parse_pragma(pragma_body)

    def add_result(self, result: Results) -> None:
        """Store a parsed result type from a pragma.

        Overrides AbstractProgramContext.add_result() to collect result type
        pragmas as structured metadata that will be attached to the circuit.

        Args:
            result: The parsed result IR object (e.g., Expectation, Probability, Sample).

        Raises:
            NotImplementedError: If the pragma is an adjoint_gradient result type,
                which is not supported by this pipeline.
        """
        raw_pragma = f"#pragma {self._last_pragma_command}"
        if "adjoint_gradient" in self._last_pragma_command:
            raise NotImplementedError(
                "adjoint_gradient result type is not supported in the Qiskit compilation pipeline. "
            )
        self._result_types.append({"raw_pragma": raw_pragma, "parsed": result})
        qc = self._circuit_stack[0]
        if qc.metadata is None:
            qc.metadata = {}
        qc.metadata["braket_result_pragmas"] = self._result_types

    def _push_scoped_circuit(self) -> QuantumCircuit:
        """Push an empty body circuit onto the stack that shares the parent's bit objects."""
        body = self._active_circuit.copy_empty_like()
        self._circuit_stack.append(body)
        return body

    def _extend_bits(self, target: QuantumCircuit, source: QuantumCircuit) -> None:
        """Add to target any bits that source has beyond target's current bit list."""
        target.add_bits(source.qubits[target.num_qubits :])
        target.add_bits(source.clbits[target.num_clbits :])

    def _ensure_qubit_capacity(self, target: Sequence[int]) -> None:
        """Grow the active circuit so it can address every index in target."""
        active = self._active_circuit
        max_qubits = (max(target) + 1) if target else -1
        num_missing_qubits = max_qubits - active.num_qubits
        active.add_bits([Qubit() for _ in range(num_missing_qubits)])
        self.num_qubits = max(self.num_qubits, active.num_qubits)

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
        self._ensure_qubit_capacity(target)

        active.append(CircuitInstruction(gate, target))

    def handle_parameter_value(self, value: Number | Expr) -> Number | Parameter:
        return _sympy_to_qiskit(value, self._param_map) if isinstance(value, Expr) else value

    def add_measure(
        self,
        target: tuple[int, ...],
        classical_targets: Sequence[int] | None = None,
        *,
        classical_destination: Identifier | IndexedIdentifier | None = None,
    ) -> None:
        if classical_destination is None:
            self._ensure_qubit_capacity(target)
            self._add_measure_into_loose_clbits(target)
            return
        name = get_identifier_name(classical_destination)
        if name not in self._clbit_offset:
            raise ValueError(f"Classical bit register {name!r} is not declared")
        self._ensure_qubit_capacity(target)
        self._add_measure_into_register(target, classical_targets, self._clbit_offset[name])

    def _add_measure_into_loose_clbits(self, target: tuple[int, ...]) -> None:
        """Measure without a classical destination, synthesizing loose clbits to receive the results."""
        active = self._active_circuit
        if active.num_clbits < len(target):
            active.add_bits([Clbit() for _ in range(len(target) - active.num_clbits)])
        for idx, qubit in enumerate(target):
            active.measure(qubit, idx)

    def _add_measure_into_register(
        self,
        target: tuple[int, ...],
        classical_targets: Sequence[int] | None,
        offset: int,
    ) -> None:
        """Measure into a declared bit register at the given flat-clbit offset."""
        active = self._active_circuit
        for idx, qubit in enumerate(target):
            local_index = classical_targets[idx] if classical_targets else idx
            active.measure(qubit, offset + local_index)

    def add_verbatim_marker(self, marker: VerbatimBoxDelimiter) -> None:
        """Handle verbatim box start/end markers.

        When START_VERBATIM is received:
        - Push a scoped body circuit onto the stack that shares the parent's bits
        - Set _in_verbatim_box flag to True

        When END_VERBATIM is received:
        - Pop the scoped body and propagate any new qubits/clbits back to the parent
        - Wrap the body in a BoxOp and append it to the parent circuit
        - Reset verbatim state

        Args:
            marker: VerbatimBoxDelimiter indicating START_VERBATIM or END_VERBATIM

        Raises:
            ValueError: On nested verbatim boxes, an unmatched END_VERBATIM, or an invalid marker
        """

        if marker == VerbatimBoxDelimiter.START_VERBATIM:
            if self._in_verbatim_box:
                raise ValueError("Nested verbatim boxes are not supported")

            self._push_scoped_circuit()
            self._in_verbatim_box = True

        elif marker == VerbatimBoxDelimiter.END_VERBATIM:
            if not self._in_verbatim_box:
                raise ValueError("Verbatim box end marker without matching start")

            body = self._circuit_stack.pop()
            parent = self._active_circuit
            self._extend_bits(parent, body)
            parent.append(BoxOp(body, label=self._verbatim_box_name), parent.qubits, parent.clbits)
            self._in_verbatim_box = False

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

        true_body = self._push_scoped_circuit()
        yield True
        self._circuit_stack.pop()

        false_body = self._push_scoped_circuit()
        yield False
        self._circuit_stack.pop()

        actual_false = false_body if false_body.data else None

        if not true_body.data and not actual_false:
            raise ValueError(
                "Branching statement conditioned on a measurement has empty bodies. "
                "Both if and else branches contain no quantum operations."
            )

        # Sync parent and both bodies to the same bit layout (IfElseOp requires it).
        self._extend_bits(main, true_body)
        self._extend_bits(main, false_body)
        self._extend_bits(true_body, main)
        self._extend_bits(false_body, main)

        if_else_op = IfElseOp(resolved_condition, true_body, actual_false)
        main.append(if_else_op, main.qubits, main.clbits)

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
        probe = self._push_scoped_circuit()
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
        body = self._push_scoped_circuit()
        with self.enter_scope():
            self.declare_variable(loop_var, loop_type, SymbolLiteral(Symbol(loop_var)))
            yield
        self._circuit_stack.pop()

        indexset = tuple(iv.value for iv in index_values)
        loop_param = self._param_map.get(loop_var) or Parameter(loop_var)
        for_op = ForLoopOp(indexset, loop_param, body)
        self._extend_bits(main, body)
        main.append(for_op, main.qubits, main.clbits)

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

        body = self._push_scoped_circuit()
        yield True
        self._circuit_stack.pop()

        if not body.data:
            raise ValueError(
                "While loop conditioned on a measurement has an empty body. "
                "This would result in an infinite loop."
            )

        self._extend_bits(main, body)
        while_op = WhileLoopOp(resolved_condition, body)
        main.append(while_op, main.qubits, main.clbits)

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
        return (self._active_circuit.clbits[clbit_index], int(value))

    def _resolve_condition_from_identifier(
        self, condition: Identifier | IndexExpression
    ) -> tuple[Clbit, int]:
        """Convert a bare identifier condition (e.g., `c` or `c[0]`) to (Clbit, 1)."""
        clbit_index = self._resolve_clbit_index(condition)
        return (self._active_circuit.clbits[clbit_index], 1)

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
