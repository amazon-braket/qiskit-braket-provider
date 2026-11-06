"""End-to-end tests for declaring OpenQASM output registers and reading them back."""

import uuid

import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from braket.task_result import TaskMetadata
from braket.tasks import GateModelQuantumTaskResult
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit_braket_provider import (
    BraketLocalBackend,
    BraketQuantumTask,
    compile_to_oq3,
    to_oq3,
)


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


def test_compile_to_oq3_merges_existing_outputs_and_keeps_input_metadata() -> None:
    """Outputs already in the metadata stay outputs, and the input circuit is not modified."""
    qreg = QuantumRegister(3, "q")
    a, c, d = ClassicalRegister(1, "a"), ClassicalRegister(1, "c"), ClassicalRegister(1, "d")
    circuit = QuantumCircuit(qreg, a, c, d)
    circuit.measure(qreg, [a[0], c[0], d[0]])
    circuit.metadata = {"braket_output_variables": {"a": None}, "other": 1}

    for source in (
        to_oq3(circuit, output_registers=["c"]),
        compile_to_oq3(circuit, output_registers=["c"]),
    ):
        assert source.splitlines()[1:4] == ["output bit[1] a;", "output bit[1] c;", "bit[1] b;"]
    assert circuit.metadata == {"braket_output_variables": {"a": None}, "other": 1}


def test_result_from_outputs_bit_order_header_and_unwritten_bits() -> None:
    """Braket outputs map to Qiskit counts, memory and header without a simulator."""
    braket_result = GateModelQuantumTaskResult(
        task_metadata=TaskMetadata(id=str(uuid.uuid4()), deviceId="device", shots=3),
        additional_metadata=None,
        outputs=[
            {"res": [1, 0], "flag": [None]},  # flag was never written, so it counts as 0
            {"res": [0, 1], "flag": [True]},
            {"res": [0, 1], "flag": [1]},
        ],
    )
    task = LocalQuantumTask(braket_result)

    result = BraketQuantumTask(
        task_id=task.id, tasks=[task], backend=BraketLocalBackend(), shots=3
    ).result()

    assert result.results[0].header == {
        "creg_sizes": [["res", 2], ["flag", 1]],
        "memory_slots": 3,
    }
    # Qiskit bit order: last register first, and within a register the highest index first.
    assert result.get_memory() == ["0 01", "1 10", "1 10"]
    assert result.get_counts() == {"0 01": 1, "1 10": 2}


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


def test_backend_run_unassigned_output_bit_is_zero() -> None:
    """The simulator reports None for an output bit nothing writes to, which counts as 0."""
    circuit = QuantumCircuit(
        QuantumRegister(1, "q"), ClassicalRegister(1, "m"), ClassicalRegister(1, "u")
    )
    circuit.x(0)
    circuit.measure(0, 0)  # u is never written

    result = BraketLocalBackend().run(circuit, shots=10, output_registers=["m", "u"]).result()

    assert result.get_counts() == {"0 1": 10}


def test_output_registers_unknown_register_raises() -> None:
    """Naming a register the circuit does not have raises."""
    circuit = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "c"))
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match=r"no classical register named \['nope'\]"):
        compile_to_oq3(circuit, output_registers=["nope"])


def test_output_registers_string_raises() -> None:
    """A single string would be split into characters, so it is rejected."""
    circuit = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "res"))
    circuit.measure(0, 0)

    with pytest.raises(TypeError, match=r"not a string\. Use \['res'\]"):
        compile_to_oq3(circuit, output_registers="res")
