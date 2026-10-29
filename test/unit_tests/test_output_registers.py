"""End-to-end tests for declaring OpenQASM output registers and reading them back."""

import uuid
from collections import Counter
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from braket.ir.openqasm import Program
from braket.task_result import TaskMetadata
from braket.tasks import GateModelQuantumTaskResult
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit_braket_provider import BraketLocalBackend, BraketQuantumTask, compile_to_oq3


def test_output_registers_compile_run_and_result() -> None:
    """Outputs are declared in OpenQASM, and Qiskit counts are built from Braket outputs."""
    qreg = QuantumRegister(4, "q")
    res = ClassicalRegister(2, "res")
    flag = ClassicalRegister(1, "flag")
    aux = ClassicalRegister(1, "aux")
    circuit = QuantumCircuit(qreg, res, flag, aux)
    circuit.x(0)
    circuit.x(2)
    circuit.measure([0, 1], res)
    circuit.measure(2, flag[0])
    circuit.measure(3, aux[0])

    source = compile_to_oq3(circuit, output_registers=["res", "flag"])

    assert source == (
        "OPENQASM 3.0;\n"
        "output bit[2] res;\n"
        "output bit[1] flag;\n"
        "bit[1] b;\n"
        "qubit[4] q;\n"
        "x q[0];\n"
        "x q[2];\n"
        "res[0] = measure q[0];\n"
        "res[1] = measure q[1];\n"
        "flag[0] = measure q[2];\n"
        "b[0] = measure q[3];"
    )

    braket_result = GateModelQuantumTaskResult(
        task_metadata=TaskMetadata(id=str(uuid.uuid4()), deviceId="device", shots=3),
        additional_metadata=None,
        measurements=np.array([[0, 0, 0, 0]] * 3),
        measured_qubits=[0, 1, 2, 3],
        measurement_counts=Counter({"0000": 3}),
        outputs=[
            {"res": [True, False], "flag": [True]},
            {"res": [1, 0], "flag": [1]},
            {"res": [0, 0], "flag": [0]},
        ],
    )
    device = Mock()
    device.run.return_value = LocalQuantumTask(braket_result)

    braket_task = device.run(Program(source=source), shots=3)
    result = BraketQuantumTask(
        task_id=braket_task.id,
        tasks=[braket_task],
        backend=BraketLocalBackend(),
        shots=3,
    ).result()

    assert device.run.call_args.args[0].source == source
    # Only the output registers are reported, named in the header. aux is not reported.
    assert result.results[0].header["creg_sizes"] == [["res", 2], ["flag", 1]]
    assert result.results[0].header["memory_slots"] == 3
    # Qiskit bit order: last register first, "flag[0] res[1]res[0]".
    assert result.get_counts() == {"1 01": 2, "0 00": 1}
    assert result.get_memory() == ["1 01", "1 01", "0 00"]


def test_output_registers_unknown_register_raises() -> None:
    """Naming a register the circuit does not have raises."""
    circuit = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "c"))
    circuit.measure(0, 0)

    with pytest.raises(ValueError, match=r"no classical register named \['nope'\]"):
        compile_to_oq3(circuit, output_registers=["nope"])
