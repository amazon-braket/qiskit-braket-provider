"""Tests for verbatim pragma support in Qiskit to Braket adapter."""

from typing import Any

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import BoxOp, CircuitInstruction

from braket.circuits import Circuit
from braket.default_simulator.openqasm.interpreter import VerbatimBoxDelimiter
from braket.default_simulator.openqasm.parser.openqasm_ast import BitType, Identifier
from braket.ir.openqasm import Program
from qiskit_braket_provider import to_qiskit
from qiskit_braket_provider.providers.adapter import _QiskitProgramContext


def _get_box_ops(qiskit_circuit: QuantumCircuit) -> list[CircuitInstruction]:
    return [instr for instr in qiskit_circuit.data if isinstance(instr.operation, BoxOp)]


def _get_non_box_gates(qiskit_circuit: QuantumCircuit) -> list[CircuitInstruction]:
    return [instr for instr in qiskit_circuit.data if not isinstance(instr.operation, BoxOp)]


@pytest.mark.parametrize(
    "qasm, num_qubits, label, expected_body_gates",
    [
        (
            """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
    cnot $0, $1;
}
""",
            2,
            "verbatim",
            ["h", "cx"],
        ),
        (
            """
OPENQASM 3.0;
#pragma braket verbatim
box {
}
""",
            0,
            "verbatim",
            [],
        ),
        (
            """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
}
""",
            1,
            "custom_verbatim",
            ["h"],
        ),
    ],
    ids=["single_box_with_gates", "empty_box", "custom_label"],
)
def test_single_verbatim_box(
    qasm: str, num_qubits: int, label: str, expected_body_gates: list[str]
) -> None:
    kwargs: dict[str, Any] = {"verbatim_box_name": label} if label != "verbatim" else {}
    qc = to_qiskit(qasm, **kwargs)

    if num_qubits:
        assert qc.num_qubits == num_qubits

    box_ops = _get_box_ops(qc)
    assert len(box_ops) == 1
    assert box_ops[0].operation.label == label
    body_gates = [d.operation.name for d in box_ops[0].operation.body.data]
    assert body_gates == expected_body_gates


def test_multiple_verbatim_boxes() -> None:
    qasm = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
}
x $1;
#pragma braket verbatim
box {
    cnot $0, $1;
}
"""
    qc = to_qiskit(qasm)
    box_ops = _get_box_ops(qc)

    assert len(box_ops) == 2
    assert all(b.operation.label == "verbatim" for b in box_ops)
    assert box_ops[0].operation.body.data[0].operation.name == "h"
    assert box_ops[1].operation.body.data[0].operation.name == "cx"

    non_box = _get_non_box_gates(qc)
    assert len(non_box) == 1
    assert non_box[0].operation.name == "x"


def test_gates_outside_verbatim_box() -> None:
    qasm = """
OPENQASM 3.0;
h $0;
#pragma braket verbatim
box {
    cnot $0, $1;
}
x $1;
"""
    qc = to_qiskit(qasm)

    non_box = _get_non_box_gates(qc)
    assert [g.operation.name for g in non_box] == ["h", "x"]

    box_ops = _get_box_ops(qc)
    assert len(box_ops) == 1
    assert box_ops[0].operation.body.data[0].operation.name == "cx"


def test_explicit_qubit_register_no_duplicate_classical_bits() -> None:
    qasm = """
OPENQASM 3.0;
bit[2] c;
qubit[2] q;
h q[0];
cnot q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""
    qc = to_qiskit(qasm)

    assert qc.num_qubits == 2
    assert qc.num_clbits == 2
    measurements = [i for i in qc.data if i.operation.name == "measure"]
    assert len(measurements) == 2


def test_verbatim_box_with_measurements() -> None:
    qasm = """
OPENQASM 3.0;
bit[2] c;
#pragma braket verbatim
box {
    h $0;
    cnot $0, $1;
}
c[0] = measure $0;
c[1] = measure $1;
"""
    qc = to_qiskit(qasm)

    assert len(_get_box_ops(qc)) == 1
    measurements = [i for i in qc.data if i.operation.name == "measure"]
    assert len(measurements) == 2


def test_verbatim_box_qubit_mapping() -> None:
    qasm = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
    cnot $1, $2;
}
"""
    qc = to_qiskit(qasm)
    box_ops = _get_box_ops(qc)
    body = box_ops[0].operation.body

    h_idx = body.find_bit(body.data[0].qubits[0]).index
    cnot_q0 = body.find_bit(body.data[1].qubits[0]).index
    cnot_q1 = body.find_bit(body.data[1].qubits[1]).index

    assert (h_idx, cnot_q0, cnot_q1) == (0, 1, 2)


def test_non_contiguous_physical_qubits() -> None:
    qasm = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $2;
    cnot $2, $5;
}
"""
    qc = to_qiskit(qasm)
    assert qc.num_qubits == 6

    box_ops = _get_box_ops(qc)
    body = box_ops[0].operation.body

    h_idx = body.find_bit(body.data[0].qubits[0]).index
    cnot_q0 = body.find_bit(body.data[1].qubits[0]).index
    cnot_q1 = body.find_bit(body.data[1].qubits[1]).index

    assert (h_idx, cnot_q0, cnot_q1) == (2, 2, 5)


def test_verbatim_box_adds_qubits_to_main_circuit() -> None:
    qasm = """
OPENQASM 3.0;
x $0;
#pragma braket verbatim
box {
    h $1;
    cnot $1, $2;
    cnot $2, $3;
}
"""
    qc = to_qiskit(qasm)
    assert qc.num_qubits >= 4

    box_ops = _get_box_ops(qc)
    assert len(box_ops) == 1
    assert box_ops[0].operation.body.num_qubits == 4


@pytest.mark.parametrize(
    "source, to_qiskit_kwargs",
    [
        (
            """
OPENQASM 3.0;
h $0;
cnot $0, $1;
""",
            {},
        ),
        (
            Circuit().h(0).cnot(0, 1),
            {"add_measurements": False},
        ),
        (
            Program(
                source="""
OPENQASM 3.0;
h $0;
cnot $0, $1;
"""
            ),
            {},
        ),
    ],
    ids=["openqasm_str", "braket_circuit", "braket_program"],
)
def test_no_verbatim_pragma(
    source: str | Circuit | Program, to_qiskit_kwargs: dict[str, Any]
) -> None:
    qc = to_qiskit(source, **to_qiskit_kwargs)
    assert len(_get_box_ops(qc)) == 0
    gate_names = [d.operation.name for d in qc.data]
    assert "h" in gate_names
    assert "cx" in gate_names


@pytest.mark.parametrize(
    "markers, error_match",
    [
        (
            [VerbatimBoxDelimiter.START_VERBATIM, VerbatimBoxDelimiter.START_VERBATIM],
            "Nested verbatim boxes are not supported",
        ),
        (
            [VerbatimBoxDelimiter.END_VERBATIM],
            "Verbatim box end marker without matching start",
        ),
        (
            ["invalid_marker"],
            "Verbatim box created using invalid marker",
        ),
    ],
    ids=["nested_start", "end_without_start", "invalid_marker"],
)
def test_context_marker_errors(markers: list[VerbatimBoxDelimiter | str], error_match: str) -> None:
    context = _QiskitProgramContext()
    context.add_qubits("q", 2)

    with pytest.raises(ValueError, match=error_match):
        for m in markers:
            context.add_verbatim_marker(m)


def test_unclosed_verbatim_box_circuit_property_error() -> None:
    context = _QiskitProgramContext()
    context.add_qubits("q", 2)
    context.add_verbatim_marker(VerbatimBoxDelimiter.START_VERBATIM)

    with pytest.raises(ValueError, match="Unclosed verbatim box at end of program"):
        _ = context.circuit


def test_unclosed_verbatim_box_syntax_error() -> None:
    qasm = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
"""
    with pytest.raises(Exception):  # noqa: B017
        to_qiskit(qasm)


def test_bit_declaration_with_identifier_size() -> None:
    qasm = """
OPENQASM 3.0;
input bit[n] alpha;
h $0;
"""
    qc = to_qiskit(qasm)
    assert qc is not None
    assert qc.num_qubits == 1
    assert qc.num_clbits == 0


def test_bit_declaration_without_size() -> None:
    qasm = """
OPENQASM 3.0;
bit c;
h $0;
"""
    qc = to_qiskit(qasm)
    assert qc.num_clbits == 1
    assert qc.num_qubits == 1


def test_bit_declaration_without_size_in_verbatim() -> None:
    qasm = """
OPENQASM 3.0;
bit c;
#pragma braket verbatim
box {
    h $0;
}
"""
    qc = to_qiskit(qasm)
    assert qc.num_clbits == 1
    box_ops = _get_box_ops(qc)
    assert len(box_ops) == 1


def test_bit_declaration_with_identifier_size_in_verbatim() -> None:
    ctx = _QiskitProgramContext()
    ctx.declare_variable("c", BitType(size=Identifier(name="n")))
    assert ctx.circuit.num_clbits == 0
