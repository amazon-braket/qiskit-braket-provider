"""End-to-end tests for result pragma support through to_qiskit → to_braket pipeline."""

import pytest
from braket.circuits import Circuit as BraketCircuit

from qiskit_braket_provider import to_braket, to_qiskit


def _to_oq3(braket_circuit: BraketCircuit) -> str:
    """Convert a Braket circuit to OpenQASM 3 source string."""
    return braket_circuit.to_ir("OPENQASM").source


def test_expectation_z_produces_measurement():
    """Expectation of Z should add measurement but no rotation gates."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation z(q[0]) @ z(q[1])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    expected = (
        "OPENQASM 3.0;\n"
        "bit[2] b;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "b[0] = measure q[0];\n"
        "b[1] = measure q[1];"
    )
    assert _to_oq3(braket_circuit) == expected


def test_expectation_x_produces_h_rotation():
    """Expectation of X should add H rotation gate before measurement."""
    source = """OPENQASM 3.0;
qubit[2] q;
cnot q[0], q[1];
#pragma braket result expectation x(q[0])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    expected = (
        "OPENQASM 3.0;\n"
        "bit[1] b;\n"
        "qubit[2] q;\n"
        "cnot q[0], q[1];\n"
        "h q[0];\n"
        "b[0] = measure q[0];"
    )
    assert _to_oq3(braket_circuit) == expected


def test_expectation_y_produces_z_s_h_rotation():
    """Expectation of Y should add Z, S, H rotation gates before measurement."""
    source = """OPENQASM 3.0;
qubit[2] q;
cnot q[0], q[1];
#pragma braket result expectation y(q[0])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    expected = (
        "OPENQASM 3.0;\n"
        "bit[1] b;\n"
        "qubit[2] q;\n"
        "cnot q[0], q[1];\n"
        "z q[0];\n"
        "s q[0];\n"
        "h q[0];\n"
        "b[0] = measure q[0];"
    )
    assert _to_oq3(braket_circuit) == expected


def test_probability_no_rotation():
    """Probability result type needs measurement but no rotation."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result probability q[0], q[1]
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    expected = (
        "OPENQASM 3.0;\n"
        "bit[2] b;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "b[0] = measure q[0];\n"
        "b[1] = measure q[1];"
    )
    assert _to_oq3(braket_circuit) == expected


def test_tensor_product_x_y():
    """Tensor product x @ y should apply per-qubit rotations."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result expectation x(q[0]) @ y(q[1])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    expected = (
        "OPENQASM 3.0;\n"
        "bit[2] b;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "h q[0];\n"
        "z q[1];\n"
        "s q[1];\n"
        "h q[1];\n"
        "b[0] = measure q[0];\n"
        "b[1] = measure q[1];"
    )
    assert _to_oq3(braket_circuit) == expected


def test_state_vector_no_rotation_instructions():
    """StateVector result type should not produce rotation gate instructions.

    Note: The Braket SDK's to_ir() serialization adds measure_all() to the OQ3
    output when no explicit measurements exist. This is SDK behavior.
    We verify at the Circuit instruction level that no rotations or measurements
    were added by our pass.
    """
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result state_vector
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    gate_names = [instr.operator.name for instr in braket_circuit.instructions]
    assert gate_names == ["H", "CNot"]


def test_multiple_result_types():
    """Multiple result pragmas should produce combined rotations."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result expectation x(q[0])
#pragma braket result variance y(q[1])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    expected = (
        "OPENQASM 3.0;\n"
        "bit[2] b;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "h q[0];\n"
        "z q[1];\n"
        "s q[1];\n"
        "h q[1];\n"
        "b[0] = measure q[0];\n"
        "b[1] = measure q[1];"
    )
    assert _to_oq3(braket_circuit) == expected


def test_from_openqasm_string_directly():
    """to_braket should handle OQ3 string with result pragmas end-to-end."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation z(q[0]) @ z(q[1])
"""
    braket_circuit = to_braket(source)

    expected = (
        "OPENQASM 3.0;\n"
        "bit[2] b;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "b[0] = measure q[0];\n"
        "b[1] = measure q[1];"
    )
    assert _to_oq3(braket_circuit) == expected


def test_sample_with_observable():
    """Sample with observable should add rotation gates and measurement."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result sample x(q[0])
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    expected = (
        "OPENQASM 3.0;\n"
        "bit[1] b;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "h q[0];\n"
        "b[0] = measure q[0];"
    )
    assert _to_oq3(braket_circuit) == expected


def test_metadata_expectation_z_tensor():
    """Metadata should store raw pragma and parsed Expectation for tensor product."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation z(q[0]) @ z(q[1])
"""
    qc = to_qiskit(source)
    pragmas = qc.metadata["braket_result_pragmas"]

    assert len(pragmas) == 1
    assert pragmas[0]["raw_pragma"] == "#pragma braket result expectation z(q[0]) @ z(q[1])"
    parsed = pragmas[0]["parsed"]
    assert parsed.observable == ["z", "z"]
    assert parsed.targets == [0, 1]


def test_metadata_expectation_x():
    """Metadata should store raw pragma and parsed Expectation for single observable."""
    source = """OPENQASM 3.0;
qubit[2] q;
cnot q[0], q[1];
#pragma braket result expectation x(q[0])
"""
    qc = to_qiskit(source)
    pragmas = qc.metadata["braket_result_pragmas"]

    assert len(pragmas) == 1
    assert pragmas[0]["raw_pragma"] == "#pragma braket result expectation x(q[0])"
    parsed = pragmas[0]["parsed"]
    assert parsed.observable == ["x"]
    assert parsed.targets == [0]


def test_metadata_probability():
    """Metadata should store raw pragma and parsed Probability."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result probability q[0], q[1]
"""
    qc = to_qiskit(source)
    pragmas = qc.metadata["braket_result_pragmas"]

    assert len(pragmas) == 1
    assert pragmas[0]["raw_pragma"] == "#pragma braket result probability q[0], q[1]"
    parsed = pragmas[0]["parsed"]
    assert parsed.targets == [0, 1]


def test_metadata_sample():
    """Metadata should store raw pragma and parsed Sample."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result sample x(q[0])
"""
    qc = to_qiskit(source)
    pragmas = qc.metadata["braket_result_pragmas"]

    assert len(pragmas) == 1
    assert pragmas[0]["raw_pragma"] == "#pragma braket result sample x(q[0])"
    parsed = pragmas[0]["parsed"]
    assert parsed.observable == ["x"]
    assert parsed.targets == [0]


def test_metadata_state_vector():
    """Metadata should store raw pragma and parsed StateVector."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result state_vector
"""
    qc = to_qiskit(source)
    pragmas = qc.metadata["braket_result_pragmas"]

    assert len(pragmas) == 1
    assert pragmas[0]["raw_pragma"] == "#pragma braket result state_vector"


def test_metadata_multiple_result_types():
    """Metadata should store all result pragmas in order."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result expectation x(q[0])
#pragma braket result variance y(q[1])
"""
    qc = to_qiskit(source)
    pragmas = qc.metadata["braket_result_pragmas"]

    assert len(pragmas) == 2
    assert pragmas[0]["raw_pragma"] == "#pragma braket result expectation x(q[0])"
    assert pragmas[0]["parsed"].observable == ["x"]
    assert pragmas[0]["parsed"].targets == [0]
    assert pragmas[1]["raw_pragma"] == "#pragma braket result variance y(q[1])"
    assert pragmas[1]["parsed"].observable == ["y"]
    assert pragmas[1]["parsed"].targets == [1]


def test_hermitian_observable_produces_unitary_rotation():
    """Hermitian observable should produce a unitary rotation gate before measurement."""
    source = """OPENQASM 3.0;
qubit[1] q;
h q[0];
#pragma braket result expectation hermitian([[0, -1im], [1im, 0]]) q[0]
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    gate_names = [instr.operator.name for instr in braket_circuit.instructions]
    assert "H" in gate_names
    assert "Unitary" in gate_names
    assert "Measure" in gate_names


def test_hermitian_observable_metadata():
    """Hermitian observable should store matrix in parsed metadata."""
    source = """OPENQASM 3.0;
qubit[1] q;
h q[0];
#pragma braket result expectation hermitian([[0, -1im], [1im, 0]]) q[0]
"""
    qc = to_qiskit(source)
    pragmas = qc.metadata["braket_result_pragmas"]

    assert len(pragmas) == 1
    assert "hermitian" in pragmas[0]["raw_pragma"]
    parsed = pragmas[0]["parsed"]
    assert parsed.targets == [0]
    assert isinstance(parsed.observable[0], list)


def test_hermitian_observable_produces_unitary_rotation():
    """Hermitian observable should produce a unitary rotation gate before measurement."""
    source = """OPENQASM 3.0;
qubit[1] q;
h q[0];
#pragma braket result expectation hermitian([[0, -1im], [1im, 0]]) q[0]
"""
    qc = to_qiskit(source)
    braket_circuit = to_braket(qc)

    gate_names = [instr.operator.name for instr in braket_circuit.instructions]
    assert "H" in gate_names
    assert "Unitary" in gate_names
    assert "Measure" in gate_names


def test_hermitian_observable_metadata():
    """Hermitian observable should store matrix in parsed metadata."""
    source = """OPENQASM 3.0;
qubit[1] q;
h q[0];
#pragma braket result expectation hermitian([[0, -1im], [1im, 0]]) q[0]
"""
    qc = to_qiskit(source)
    pragmas = qc.metadata["braket_result_pragmas"]

    assert len(pragmas) == 1
    assert "hermitian" in pragmas[0]["raw_pragma"]
    parsed = pragmas[0]["parsed"]
    assert parsed.targets == [0]
    assert isinstance(parsed.observable[0], list)
