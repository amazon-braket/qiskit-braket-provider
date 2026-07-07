"""Tests for result pragma support in to_qiskit."""

import pytest
from qiskit import QuantumCircuit

from braket.ir.openqasm import Program
from qiskit_braket_provider import to_qiskit


def test_expectation_result_type():
    """to_qiskit should handle expectation result type pragma."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation z(q[0]) @ z(q[1])
"""
    circuit = to_qiskit(source)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_probability_result_type():
    """to_qiskit should handle probability result type pragma."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result probability q[0], q[1]
"""
    circuit = to_qiskit(source)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_sample_result_type():
    """to_qiskit should handle sample result type pragma."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result sample x(q[0])
"""
    circuit = to_qiskit(source)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_variance_result_type():
    """to_qiskit should handle variance result type pragma."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result variance y(q[1])
"""
    circuit = to_qiskit(source)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_state_vector_result_type():
    """to_qiskit should handle state_vector result type pragma."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result state_vector
"""
    circuit = to_qiskit(source)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_amplitude_result_type():
    """to_qiskit should handle amplitude result type pragma."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result amplitude "00", "11"
"""
    circuit = to_qiskit(source)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_density_matrix_result_type():
    """to_qiskit should handle density_matrix result type pragma."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result density_matrix q[0], q[1]
"""
    circuit = to_qiskit(source)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_multiple_result_types():
    """to_qiskit should handle multiple result type pragmas."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation x(q[0]) @ y(q[1])
#pragma braket result variance z(q[0])
"""
    circuit = to_qiskit(source)
    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2


def test_metadata_contains_result_pragmas_key():
    """Circuit metadata should contain braket_result_pragmas key."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation z(q[0])
"""
    circuit = to_qiskit(source)
    assert "braket_result_pragmas" in circuit.metadata


def test_metadata_has_correct_count():
    """Metadata should contain one entry per result pragma."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation x(q[0])
#pragma braket result variance z(q[1])
"""
    circuit = to_qiskit(source)
    assert len(circuit.metadata["braket_result_pragmas"]) == 2


def test_metadata_contains_raw_pragma():
    """Each metadata entry should contain the raw pragma string."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result expectation z(q[0])
"""
    circuit = to_qiskit(source)
    pragmas = circuit.metadata["braket_result_pragmas"]
    assert len(pragmas) == 1
    assert pragmas[0]["raw_pragma"] == "#pragma braket result expectation z(q[0])"


def test_metadata_contains_parsed_result():
    """Each metadata entry should contain the parsed IR result object."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result expectation z(q[0])
"""
    circuit = to_qiskit(source)
    pragmas = circuit.metadata["braket_result_pragmas"]
    assert len(pragmas) == 1
    parsed = pragmas[0]["parsed"]
    assert parsed is not None
    assert hasattr(parsed, "observable")
    assert hasattr(parsed, "targets")


def test_no_metadata_when_no_result_pragmas():
    """Circuit should not have braket_result_pragmas in metadata if no result pragmas."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
"""
    circuit = to_qiskit(source)
    assert circuit.metadata is None or "braket_result_pragmas" not in (circuit.metadata or {})


def test_program_object_with_result_pragmas():
    """to_qiskit should work with Program objects containing result pragmas."""
    program = Program(
        source="""
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result probability q[0], q[1]
"""
    )
    circuit = to_qiskit(program)
    assert "braket_result_pragmas" in circuit.metadata
    assert len(circuit.metadata["braket_result_pragmas"]) == 1


def test_no_measure_ops_with_result_pragmas():
    """Circuit should have no measure operations when result pragmas are present."""
    source = """
OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation z(q[0]) @ z(q[1])
"""
    circuit = to_qiskit(source)
    measure_ops = [instr for instr in circuit.data if instr.operation.name == "measure"]
    assert len(measure_ops) == 0


def test_result_type_with_physical_qubits():
    """Result pragmas should work with physical qubit notation."""
    source = """
OPENQASM 3.0;
h $0;
cnot $0, $1;
#pragma braket result expectation z all
"""
    circuit = to_qiskit(source)
    assert isinstance(circuit, QuantumCircuit)
    assert "braket_result_pragmas" in circuit.metadata


def test_adjoint_gradient_raises_not_implemented():
    """adjoint_gradient result type should raise NotImplementedError."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result adjoint_gradient expectation(z(q[0]) @ z(q[1])) all
"""
    with pytest.raises(NotImplementedError, match="adjoint_gradient"):
        to_qiskit(source)
