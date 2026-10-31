"""End-to-end tests for declaring OpenQASM output registers and reading them back."""

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qiskit_braket_provider import BraketLocalBackend, compile_to_oq3


def _circuit_with_outputs() -> QuantumCircuit:
    """Two output registers, res and flag, and one register that isn't an output, aux."""
    qreg = QuantumRegister(4, "q")
    res = ClassicalRegister(2, "res")
    flag = ClassicalRegister(1, "flag")
    aux = ClassicalRegister(1, "aux")
    circuit = QuantumCircuit(qreg, res, flag, aux)
    circuit.x(0)
    circuit.x(2)
    circuit.h(3)  # aux is random, so the counts only stay fixed if aux is left out
    circuit.measure([0, 1], res)
    circuit.measure(2, flag[0])
    circuit.measure(3, aux[0])
    return circuit


def test_compile_to_oq3_declares_output_registers() -> None:
    """Named registers are declared output, and other bits merge into the plain register b."""
    source = compile_to_oq3(_circuit_with_outputs(), output_registers=["res", "flag"])

    assert source == (
        "OPENQASM 3.0;\n"
        "output bit[2] res;\n"
        "output bit[1] flag;\n"
        "bit[1] b;\n"
        "qubit[4] q;\n"
        "x q[0];\n"
        "x q[2];\n"
        "h q[3];\n"
        "res[0] = measure q[0];\n"
        "res[1] = measure q[1];\n"
        "flag[0] = measure q[2];\n"
        "b[0] = measure q[3];"
    )


def test_backend_run_reports_output_registers() -> None:
    """The local simulator returns outputs, which become named registers in the Qiskit result."""
    result = (
        BraketLocalBackend()
        .run(_circuit_with_outputs(), shots=100, output_registers=["res", "flag"])
        .result()
    )

    assert result.results[0].header == {
        "creg_sizes": [["res", 2], ["flag", 1]],
        "memory_slots": 3,
    }
    # Qiskit bit order: last register first, "flag[0] res[1]res[0]".
    assert result.get_counts() == {"1 01": 100}
    assert result.get_memory() == ["1 01"] * 100


def test_output_registers_unknown_register_raises() -> None:
    """Naming a register the circuit does not have raises."""
    circuit = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "c"))
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match=r"no classical register named \['nope'\]"):
        compile_to_oq3(circuit, output_registers=["nope"])
