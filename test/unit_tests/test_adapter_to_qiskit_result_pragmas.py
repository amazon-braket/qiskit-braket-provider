"""Tests for result pragma support in to_qiskit."""

import pytest
from qiskit import QuantumCircuit

from braket.ir.jaqcd import AdjointGradient, Expectation, Probability, Variance
from braket.ir.openqasm import Program
from qiskit_braket_provider import to_qiskit
from qiskit_braket_provider.providers.qasm_context import _QiskitProgramContext


@pytest.mark.parametrize(
    "pragma_line",
    [
        "#pragma braket result expectation z(q[0]) @ z(q[1])",
        "#pragma braket result probability q[0], q[1]",
        "#pragma braket result sample x(q[0])",
        "#pragma braket result variance y(q[1])",
        "#pragma braket result state_vector",
        '#pragma braket result amplitude "00", "11"',
        "#pragma braket result density_matrix q[0], q[1]",
    ],
    ids=[
        "expectation",
        "probability",
        "sample",
        "variance",
        "state_vector",
        "amplitude",
        "density_matrix",
    ],
)
def test_single_result_type_parsed_to_metadata(pragma_line: str):
    """Each result type pragma should parse without error and store in metadata."""
    source = f"""OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
{pragma_line}
"""
    circuit = to_qiskit(source)

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 2
    assert "braket_result_pragmas" in circuit.metadata
    pragmas = circuit.metadata["braket_result_pragmas"]
    assert len(pragmas) == 1


def test_multiple_result_types():
    """Multiple result type pragmas should all be stored in metadata."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation x(q[0]) @ y(q[1])
#pragma braket result variance z(q[0])
"""
    circuit = to_qiskit(source)

    assert isinstance(circuit, QuantumCircuit)
    pragmas = circuit.metadata["braket_result_pragmas"]
    assert len(pragmas) == 2
    assert isinstance(pragmas[0], Expectation)
    assert isinstance(pragmas[1], Variance)


def test_metadata_contains_parsed_result_with_observable_and_targets():
    """Parsed result should contain observable and targets attributes."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
#pragma braket result expectation z(q[0])
"""
    circuit = to_qiskit(source)
    pragmas = circuit.metadata["braket_result_pragmas"]

    assert len(pragmas) == 1
    assert pragmas[0].observable == ["z"]
    assert pragmas[0].targets == [0]


def test_no_metadata_when_no_result_pragmas():
    """Circuit should not have braket_result_pragmas in metadata if no result pragmas."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
"""
    circuit = to_qiskit(source)
    assert circuit.metadata is None or "braket_result_pragmas" not in (circuit.metadata or {})


def test_program_object_with_result_pragmas():
    """to_qiskit should work with Program objects containing result pragmas."""
    program = Program(
        source="""OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result probability q[0], q[1]
"""
    )
    circuit = to_qiskit(program)
    assert "braket_result_pragmas" in circuit.metadata
    pragmas = circuit.metadata["braket_result_pragmas"]
    assert len(pragmas) == 1
    assert isinstance(pragmas[0], Probability)


def test_result_pragma_does_not_add_measurements():
    """Parsing result pragmas should not add measurement operations to the circuit."""
    source = """OPENQASM 3.0;
qubit[2] q;
h q[0];
cnot q[0], q[1];
#pragma braket result expectation z(q[0]) @ z(q[1])
"""
    circuit = to_qiskit(source)
    measure_ops = [instr for instr in circuit.data if instr.operation.name == "measure"]
    assert len(measure_ops) == 0


def test_result_type_with_physical_qubits():
    """Result pragmas should work with physical qubit notation and expand targets."""
    source = """OPENQASM 3.0;
h $0;
cnot $0, $1;
#pragma braket result expectation z($0) @ z($1)
"""
    circuit = to_qiskit(source)

    assert "braket_result_pragmas" in circuit.metadata
    pragmas = circuit.metadata["braket_result_pragmas"]
    assert pragmas[0].targets == [0, 1]


def test_adjoint_gradient_raises_not_implemented():
    """AdjointGradient result type should raise NotImplementedError."""
    ctx = _QiskitProgramContext()
    result = AdjointGradient(observable=["z"], targets=[[0]], parameters=["alpha"])
    with pytest.raises(NotImplementedError, match="AdjointGradient"):
        ctx.add_result(result)
