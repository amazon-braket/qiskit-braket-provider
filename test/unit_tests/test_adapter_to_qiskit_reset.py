"""Tests for OpenQASM ``reset`` in to_qiskit.

Reset is non-unitary, so these tests compare the structure of the resulting
QuantumCircuit against an expected circuit instead of using Operator or Statevector.
"""

from collections.abc import Sequence

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Clbit, ForLoopOp, IfElseOp, Parameter, Qubit

from braket.ir.openqasm import Program
from qiskit_braket_provider import to_qiskit
from qiskit_braket_provider.providers.braket_annotations import BraketVerbatimBox


def _single_qubit() -> QuantumCircuit:
    expected = QuantumCircuit(1)
    expected.reset(0)
    return expected


def _register() -> QuantumCircuit:
    expected = QuantumCircuit(3)
    expected.x([0, 1, 2])
    expected.reset([0, 1, 2])
    return expected


def _register_slice() -> QuantumCircuit:
    expected = QuantumCircuit(3)
    expected.reset([1, 2])
    return expected


def _mid_circuit() -> QuantumCircuit:
    expected = QuantumCircuit(2)
    expected.h(0)
    expected.cx(0, 1)
    expected.reset(1)
    expected.x(1)
    return expected


def _physical_qubit() -> QuantumCircuit:
    expected = QuantumCircuit([Qubit() for _ in range(4)])
    expected.x(0)
    expected.reset(3)
    return expected


@pytest.mark.parametrize(
    "qasm, expected",
    [
        ("qubit[1] q;\nreset q[0];", _single_qubit()),
        ("qubit[3] q;\nx q;\nreset q;", _register()),
        ("qubit[3] q;\nreset q[1:2];", _register_slice()),
        ("qubit[2] q;\nh q[0];\ncnot q[0], q[1];\nreset q[1];\nx q[1];", _mid_circuit()),
        ("x $0;\nreset $3;", _physical_qubit()),
    ],
    ids=["single_qubit", "register", "register_slice", "mid_circuit", "physical_qubit"],
)
def test_reset(qasm: str, expected: QuantumCircuit):
    assert to_qiskit(f"OPENQASM 3.0;\n{qasm}") == expected


@pytest.mark.parametrize(
    "labels, qasm, expected_reset_index",
    [
        ((1, 2, 5), "x $1;\nreset $5;", 2),
        ((1, 2, 5), "x $1;\nreset $2;", 1),
    ],
    ids=["top_label", "middle_label"],
)
def test_reset_physical_qubit_translated_through_labels(
    labels: Sequence[int], qasm: str, expected_reset_index: int
):
    qc = to_qiskit(Program(source=f"OPENQASM 3.0;\n{qasm}"), physical_qubit_labels=labels)

    expected = QuantumCircuit([Qubit() for _ in range(expected_reset_index + 1)])
    expected.x(0)
    expected.reset(expected_reset_index)
    assert qc == expected


def test_reset_physical_qubit_off_device_raises():
    with pytest.raises(ValueError, match=r"Physical qubit \$4 is not on the target device"):
        to_qiskit("OPENQASM 3.0;\nreset $4;", physical_qubit_labels=(1, 2, 5))


def test_reset_inside_if_else():
    qasm = """
OPENQASM 3.0;
qubit[2] q;
bit[1] c;
c[0] = measure q[0];
if (c[0]) {
    reset q[1];
} else {
    x q[1];
}
"""
    qc = to_qiskit(qasm)

    expected = QuantumCircuit(2)
    expected.add_bits([Clbit()])
    expected.measure(0, 0)
    true_body = expected.copy_empty_like()
    true_body.reset(1)
    false_body = expected.copy_empty_like()
    false_body.x(1)
    expected.append(
        IfElseOp((expected.clbits[0], 1), true_body, false_body), expected.qubits, expected.clbits
    )
    assert qc == expected


def test_reset_inside_for_loop():
    qasm = """
OPENQASM 3.0;
qubit[1] q;
for int[8] i in [0:2] {
    x q[0];
    reset q[0];
}
"""
    qc = to_qiskit(qasm)

    body = QuantumCircuit(1)
    body.x(0)
    body.reset(0)
    expected = QuantumCircuit(1)
    expected.append(ForLoopOp(range(3), Parameter("i"), body), expected.qubits)
    assert qc == expected


def test_reset_only_for_loop_body_is_kept():
    """A loop body whose only operation is reset is still a quantum body, not unrolled away."""
    qc = to_qiskit("OPENQASM 3.0;\nqubit[1] q;\nfor int[8] i in [0:1] {\n    reset q[0];\n}")

    body = QuantumCircuit(1)
    body.reset(0)
    expected = QuantumCircuit(1)
    expected.append(ForLoopOp(range(2), Parameter("i"), body), expected.qubits)
    assert qc == expected


def test_reset_inside_verbatim_box():
    qasm = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    x $0;
    reset $0;
}
"""
    qc = to_qiskit(qasm)

    body = QuantumCircuit([Qubit()])
    body.x(0)
    body.reset(0)
    expected = QuantumCircuit(body.qubits)
    expected.append(BraketVerbatimBox(body), expected.qubits)
    assert qc == expected


def test_reset_inside_verbatim_box_adds_qubits_to_main_circuit():
    """A qubit first referenced by a reset inside a verbatim box widens the main circuit."""
    qc = to_qiskit("OPENQASM 3.0;\nx $0;\n#pragma braket verbatim\nbox {\n    reset $2;\n}")

    expected = QuantumCircuit([Qubit() for _ in range(3)])
    expected.x(0)
    body = expected.copy_empty_like()
    body.reset(2)
    expected.append(BraketVerbatimBox(body), expected.qubits)
    assert qc == expected
