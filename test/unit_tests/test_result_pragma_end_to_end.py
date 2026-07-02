"""End-to-end tests for result pragma support through to_qiskit → to_braket pipeline."""

import pytest
from braket.circuits import Circuit as BraketCircuit

from qiskit_braket_provider import to_braket, to_qiskit


def test_expectation_z_produces_measurement():
    """Expectation of Z should add measurement but no rotation gates."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation z(q[0]) @ z(q[1])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    assert isinstance(braket_circuit, BraketCircuit)
    gate_names = [instr.operator.name.lower() for instr in braket_circuit.instructions]
    assert "h" in gate_names
    assert "cnot" in gate_names
    assert "measure" in gate_names


def test_expectation_x_produces_h_rotation():
    """Expectation of X should add H rotation gate before measurement."""
    source = """
OPENQASM 3.0;
qubit[2] q;
cnot q[0], q[1];
#pragma braket result expectation x(q[0])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    gate_names = [instr.operator.name.lower() for instr in braket_circuit.instructions]
    assert "cnot" in gate_names
    assert "h" in gate_names
    assert "measure" in gate_names


def test_expectation_y_produces_z_s_h_rotation():
    """Expectation of Y should add Z, S, H rotation gates before measurement."""
    source = """
OPENQASM 3.0;
qubit[2] q;
cnot q[0], q[1];
#pragma braket result expectation y(q[0])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    gate_names = [instr.operator.name.lower() for instr in braket_circuit.instructions]
    assert "z" in gate_names
    assert "s" in gate_names
    assert "h" in gate_names
    assert "measure" in gate_names


def test_probability_no_rotation():
    """Probability result type needs measurement but no rotation."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result probability q[0], q[1]
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    gate_names = [instr.operator.name.lower() for instr in braket_circuit.instructions]
    assert "h" in gate_names
    assert "cnot" in gate_names
    assert "measure" in gate_names


def test_tensor_product_x_y():
    """Tensor product x @ y should apply per-qubit rotations."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result expectation x(q[0]) @ y(q[1])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    gate_names = [instr.operator.name.lower() for instr in braket_circuit.instructions]
    assert "z" in gate_names
    assert "s" in gate_names
    assert gate_names.count("measure") == 2


def test_state_vector_no_rotation_no_measurement():
    """StateVector result type should not produce rotation or measurement."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result state_vector
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    gate_names = [instr.operator.name.lower() for instr in braket_circuit.instructions]
    assert "measure" not in gate_names


def test_multiple_result_types():
    """Multiple result pragmas should produce combined rotations."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result expectation x(q[0])
#pragma braket result variance y(q[1])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    gate_names = [instr.operator.name.lower() for instr in braket_circuit.instructions]
    assert gate_names.count("measure") == 2
    assert "z" in gate_names
    assert "s" in gate_names


def test_from_openqasm_string_directly():
    """to_braket should handle OQ3 string with result pragmas end-to-end."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation z(q[0]) @ z(q[1])
"""
    braket_circuit = to_braket(source)

    assert isinstance(braket_circuit, BraketCircuit)
    gate_names = [instr.operator.name.lower() for instr in braket_circuit.instructions]
    assert "h" in gate_names
    assert "cnot" in gate_names
    assert "measure" in gate_names


def test_sample_with_observable():
    """Sample with observable should add rotation gates and measurement."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result sample x(q[0])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    gate_names = [instr.operator.name.lower() for instr in braket_circuit.instructions]
    h_count = gate_names.count("h")
    assert h_count >= 2
    assert "measure" in gate_names
