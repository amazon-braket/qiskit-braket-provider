"""Tests for the Braket-formatting transpiler passes."""

from collections.abc import Callable

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp, ClassicalRegister, Clbit, IfElseOp, QuantumRegister
from qiskit.transpiler import PassManager

from qiskit_braket_provider.providers.gate_mappings import _BRAKET_VERBATIM_BOX_NAME
from qiskit_braket_provider.providers.passes import (
    ConsolidateClbits,
    MoveMeasurementsToEnd,
    WrapInVerbatimBox,
)


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _two_cregs_circuit() -> QuantumCircuit:
    qr = QuantumRegister(2, "q")
    cr1 = ClassicalRegister(1, "a")
    cr2 = ClassicalRegister(1, "b")
    qc = QuantumCircuit(qr, cr1, cr2)
    qc.h(0)
    qc.measure(0, cr1[0])
    qc.measure(1, cr2[0])
    return qc


def _default_creg_circuit() -> QuantumCircuit:
    """QuantumCircuit(2, 2) creates a default register named 'c'."""
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])
    return qc


def _no_clbits_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _output_register_circuit() -> QuantumCircuit:
    """A circuit whose register is marked as an OpenQASM output variable."""
    creg = ClassicalRegister(2, "c")
    qc = QuantumCircuit(2)
    qc.add_register(creg)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, creg[0])
    qc.measure(1, creg[1])
    qc.metadata = {"braket_output_variables": {"c": None}}
    return qc


def _output_and_scratch_circuit() -> QuantumCircuit:
    """Output register + a plain scratch register + a loose Clbit."""
    first = ClassicalRegister(2, "first")
    scratch = ClassicalRegister(1, "scratch")
    qc = QuantumCircuit(3)
    qc.add_register(first)
    qc.add_register(scratch)
    qc.add_bits([Clbit()])
    qc.measure(0, first[0])
    qc.measure(1, first[1])
    qc.measure(2, scratch[0])
    qc.metadata = {"braket_output_variables": {"first": None}}
    return qc


def _shadow_b_circuit() -> QuantumCircuit:
    """Output register named 'b' forces plain-bit register to fall back to 'b0'."""
    b = ClassicalRegister(1, "b")
    qc = QuantumCircuit(2)
    qc.add_register(b)
    qc.add_bits([Clbit()])
    qc.measure(0, b[0])
    qc.measure(1, qc.clbits[1])
    qc.metadata = {"braket_output_variables": {"b": None}}
    return qc


def _shadow_b_and_b0_circuit() -> QuantumCircuit:
    """Output registers 'b' and 'b0' both shadow the plain-bit fallback names."""
    b = ClassicalRegister(1, "b")
    b0 = ClassicalRegister(1, "b0")
    qc = QuantumCircuit(3)
    qc.add_register(b)
    qc.add_register(b0)
    qc.add_bits([Clbit()])
    qc.measure(0, b[0])
    qc.measure(1, b0[0])
    qc.measure(2, qc.clbits[2])
    qc.metadata = {"braket_output_variables": {"b": None, "b0": None}}
    return qc


def _all_output_registers_circuit() -> QuantumCircuit:
    """Every clbit lives in an output-declared register; ``plain`` is empty."""
    a = ClassicalRegister(1, "a")
    b = ClassicalRegister(1, "b")
    qc = QuantumCircuit(2)
    qc.add_register(a)
    qc.add_register(b)
    qc.measure(0, a[0])
    qc.measure(1, b[0])
    qc.metadata = {"braket_output_variables": {"a": None, "b": None}}
    return qc


def _mid_measure_circuit() -> QuantumCircuit:
    """A circuit with a mid-circuit measurement, followed by more ops."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)
    qc.measure(1, 1)
    return qc


def _if_else_circuit_loose_clbit() -> tuple[QuantumCircuit, Clbit]:
    """Circuit with a loose Clbit used as an IfElseOp condition."""
    qc = QuantumCircuit(1)
    qc.add_bits([Clbit()])
    condition_bit = qc.clbits[0]
    qc.measure(0, condition_bit)
    with qc.if_test((condition_bit, 1)):
        qc.x(0)
    return qc, condition_bit


def _if_else_circuit_existing_creg() -> tuple[QuantumCircuit, Clbit]:
    """Circuit with an existing ClassicalRegister used as an IfElseOp condition."""
    cr = ClassicalRegister(1, "a")
    qc = QuantumCircuit(QuantumRegister(1, "q"), cr)
    condition_bit = cr[0]
    qc.measure(0, condition_bit)
    with qc.if_test((condition_bit, 1)):
        qc.x(0)
    return qc, condition_bit


def _if_else_circuit_for_verbatim() -> QuantumCircuit:
    """Circuit with an IfElseOp — used to check verbatim wrapping preserves it."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
    return qc


def _get_if_else(circuit: QuantumCircuit) -> IfElseOp:
    return next(instr.operation for instr in circuit.data if isinstance(instr.operation, IfElseOp))


@pytest.mark.parametrize(
    "build_circuit,expected_creg_names,expected_num_clbits",
    [
        (_two_cregs_circuit, ["b"], 2),
        (_default_creg_circuit, ["b"], 2),
        (_no_clbits_circuit, [], 0),
        (_output_register_circuit, ["c"], 2),
        (_output_and_scratch_circuit, ["first", "b"], 4),
        (_shadow_b_circuit, ["b", "b0"], 2),
        (_shadow_b_and_b0_circuit, ["b", "b0", "b1"], 3),
        (_all_output_registers_circuit, ["a", "b"], 2),
    ],
    ids=[
        "two_regs",
        "renames_default_reg",
        "no_clbits_noop",
        "output_register_preserved",
        "output_and_plain_bits",
        "shadow_b_falls_back_to_b0",
        "shadow_b_and_b0_falls_back_to_b1",
        "no_plain_register_when_all_outputs",
    ],
)
def test_consolidate_clbits(
    build_circuit: Callable, expected_creg_names: list[str], expected_num_clbits: int
) -> None:
    result = PassManager([ConsolidateClbits()]).run(build_circuit())
    assert [creg.name for creg in result.cregs] == expected_creg_names
    assert result.num_clbits == expected_num_clbits


@pytest.mark.parametrize(
    "build_circuit",
    [_if_else_circuit_loose_clbit, _if_else_circuit_existing_creg],
    ids=["loose_clbit", "existing_creg"],
)
def test_consolidate_clbits_preserves_if_else_condition(build_circuit: Callable) -> None:
    """The IfElseOp condition must still point at the same Clbit after consolidation."""
    circuit, condition_bit = build_circuit()
    result = PassManager([ConsolidateClbits()]).run(circuit)
    cond_clbit, cond_value = _get_if_else(result).condition
    assert cond_clbit == condition_bit
    assert cond_value == 1
    # The Clbit is now a member of the "b" register.
    assert any(condition_bit in creg for creg in result.cregs if creg.name == "b")


@pytest.mark.parametrize(
    "build_circuit,dynamic_circuits_supported,expected_op_order",
    [
        (_mid_measure_circuit, False, ["h", "cx", "measure", "measure"]),
        (_mid_measure_circuit, True, ["h", "measure", "cx", "measure"]),
        (_bell_circuit, False, ["h", "cx", "measure", "measure"]),
    ],
    ids=["reorders_mid_measure", "dynamic_circuits_noop", "already_at_end_noop"],
)
def test_move_measurements_to_end(
    build_circuit: Callable, dynamic_circuits_supported: bool, expected_op_order: list[str]
) -> None:
    result = PassManager([MoveMeasurementsToEnd(dynamic_circuits_supported)]).run(build_circuit())
    assert [instr.operation.name for instr in result.data] == expected_op_order


@pytest.mark.parametrize(
    "build_circuit,dynamic_circuits_supported,expected_inner_ops,expected_trailing_ops,"
    "expected_metadata",
    [
        (
            _bell_circuit,
            False,
            ["h", "cx"],
            ["measure", "measure"],
            {},
        ),
        (
            _bell_circuit,
            True,
            ["h", "cx", "measure", "measure"],
            [],
            {},
        ),
        (
            _output_register_circuit,
            False,
            ["h", "cx"],
            ["measure", "measure"],
            {"braket_output_variables": {"c": None}},
        ),
    ],
    ids=[
        "measurements_outside_by_default",
        "everything_inside_when_dynamic",
        "preserves_metadata",
    ],
)
def test_wrap_in_verbatim_box(
    build_circuit: Callable,
    dynamic_circuits_supported: bool,
    expected_inner_ops: list[str],
    expected_trailing_ops: list[str],
    expected_metadata: dict,
) -> None:
    result = PassManager([
        WrapInVerbatimBox(dynamic_circuits_supported=dynamic_circuits_supported)
    ]).run(build_circuit())

    top_level_boxes = [instr for instr in result.data if isinstance(instr.operation, BoxOp)]
    assert len(top_level_boxes) == 1
    assert top_level_boxes[0].operation.label == _BRAKET_VERBATIM_BOX_NAME

    op_names = [instr.operation.name for instr in result.data]
    box_idx = op_names.index("box")
    assert op_names[box_idx + 1 :] == expected_trailing_ops

    inner_ops = [instr.operation.name for instr in top_level_boxes[0].operation.blocks[0].data]
    assert inner_ops == expected_inner_ops
    assert result.metadata == expected_metadata


def test_wrap_in_verbatim_box_preserves_if_else_body_when_dynamic() -> None:
    """When dynamic_circuits_supported=True, an IfElseOp is placed inside the verbatim box."""
    result = PassManager([WrapInVerbatimBox(dynamic_circuits_supported=True)]).run(
        _if_else_circuit_for_verbatim()
    )

    top_level_boxes = [instr for instr in result.data if isinstance(instr.operation, BoxOp)]
    assert len(top_level_boxes) == 1
    inner_ops = top_level_boxes[0].operation.blocks[0].data
    assert any(isinstance(instr.operation, IfElseOp) for instr in inner_ops)


@pytest.mark.parametrize(
    "circuit",
    [QuantumCircuit(2, 2), QuantumCircuit()],
    ids=["wires_only", "nothing_at_all"],
)
def test_wrap_in_verbatim_box_empty_circuit(circuit: QuantumCircuit) -> None:
    """An empty circuit yields a single empty verbatim box spanning its wires."""
    result = PassManager([WrapInVerbatimBox()]).run(circuit)

    top_level_boxes = [instr for instr in result.data if isinstance(instr.operation, BoxOp)]
    assert len(top_level_boxes) == 1
    box_instr = top_level_boxes[0]
    assert box_instr.operation.label == _BRAKET_VERBATIM_BOX_NAME
    assert list(box_instr.qubits) == list(circuit.qubits)
    assert list(box_instr.clbits) == list(circuit.clbits)
    assert list(box_instr.operation.blocks[0].data) == []
