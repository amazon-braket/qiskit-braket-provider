"""Tests for OQ3 preparation transpiler passes (ConsolidateClbits, WrapInVerbatimBox)."""

from collections.abc import Callable

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Clbit, QuantumRegister
from qiskit.transpiler import PassManager

from qiskit_braket_provider.providers.passes import ConsolidateClbits, WrapInVerbatimBox


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


def _single_creg_circuit() -> QuantumCircuit:
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


@pytest.mark.parametrize(
    "build_circuit,expected_creg_names,expected_num_clbits",
    [
        (_two_cregs_circuit, ["b"], 2),
        (_single_creg_circuit, ["b"], 2),
        (_no_clbits_circuit, [], 0),
        (_output_register_circuit, ["c"], 2),
        (_output_and_scratch_circuit, ["first", "b"], 4),
        (_shadow_b_circuit, ["b", "b0"], 2),
        (_shadow_b_and_b0_circuit, ["b", "b0", "b1"], 3),
    ],
    ids=[
        "two_regs",
        "single_reg_noop",
        "no_clbits_noop",
        "output_register_preserved",
        "output_and_plain_bits",
        "shadow_b_falls_back_to_b0",
        "shadow_b_and_b0_falls_back_to_b1",
    ],
)
def test_consolidate_clbits(
    build_circuit: Callable, expected_creg_names: list[str], expected_num_clbits: int
) -> None:
    result = PassManager([ConsolidateClbits()]).run(build_circuit())
    assert [creg.name for creg in result.cregs] == expected_creg_names
    assert result.num_clbits == expected_num_clbits


@pytest.mark.parametrize(
    "build_circuit,label,expected_label,expected_inner_ops,expected_measures,expected_metadata",
    [
        (_bell_circuit, None, "verbatim", ["h", "cx"], 2, {}),
        (_bell_circuit, "my_label", "my_label", ["h", "cx"], 2, {}),
        (
            _output_register_circuit,
            None,
            "verbatim",
            [],
            2,
            {"braket_output_variables": {"c": None}},
        ),
    ],
    ids=["default_label", "custom_label", "preserves_metadata"],
)
def test_wrap_in_verbatim_box(
    build_circuit: Callable,
    label: str | None,
    expected_label: str,
    expected_inner_ops: list[str],
    expected_measures: int,
    expected_metadata: dict,
) -> None:
    pass_ = WrapInVerbatimBox() if label is None else WrapInVerbatimBox(label)
    result = PassManager([pass_]).run(build_circuit())

    op_names = [instr.operation.name for instr in result.data]
    assert op_names.count("box") == 1
    assert op_names.count("measure") == expected_measures

    box_op = next(instr.operation for instr in result.data if instr.operation.name == "box")
    assert box_op.label == expected_label
    inner_ops = [instr.operation.name for instr in box_op.blocks[0].data]
    assert inner_ops == expected_inner_ops
    assert result.metadata == expected_metadata
