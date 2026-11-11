"""End-to-end tests for declaring OpenQASM output registers and reading them back."""

import uuid
from collections.abc import Callable

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


def _circuit_with_unwritten_output() -> QuantumCircuit:
    """Output register u is never written, so the simulator reports it as None."""
    circuit = QuantumCircuit(
        QuantumRegister(1, "q"), ClassicalRegister(1, "m"), ClassicalRegister(1, "u")
    )
    circuit.x(0)
    circuit.measure(0, 0)
    return circuit


@pytest.mark.parametrize(
    ("make_circuit", "output_registers", "creg_sizes", "bitstring"),
    [
        # Qiskit bit order: last register first, "flag[0] res[1]res[0]". aux is not reported.
        pytest.param(
            _circuit_with_outputs, ["res", "flag"], [["res", 2], ["flag", 1]], "1 01", id="named"
        ),
        # An output bit nothing writes to counts as 0.
        pytest.param(
            _circuit_with_unwritten_output, ["m", "u"], [["m", 1], ["u", 1]], "0 1", id="unwritten"
        ),
    ],
)
def test_backend_run_reports_output_registers(
    make_circuit: Callable[[], QuantumCircuit],
    output_registers: list[str],
    creg_sizes: list[list[str | int]],
    bitstring: str,
) -> None:
    """The local simulator returns outputs, which become named registers in the Qiskit result."""
    shots = 10
    result = (
        BraketLocalBackend()
        .run(make_circuit(), shots=shots, output_registers=output_registers)
        .result()
    )

    assert result.results[0].header == {
        "creg_sizes": creg_sizes,
        "memory_slots": sum(size for _, size in creg_sizes),
    }
    assert result.get_counts() == {bitstring: shots}
    assert result.get_memory() == [bitstring] * shots


@pytest.mark.parametrize(
    ("register_name", "output_registers", "error", "match"),
    [
        pytest.param(
            "res", ["nope"], ValueError, r"no classical register named \['nope'\]", id="unknown"
        ),
        # A single string would be split into characters.
        pytest.param("res", "res", TypeError, r"not a string\. Use \['res'\]", id="string"),
        # The exporter would rename z to z_0 and drop its output declaration.
        pytest.param("z", ["z"], ValueError, r"\['z'\] clash with", id="gate_name"),
        pytest.param("measure", ["measure"], ValueError, r"\['measure'\] clash", id="keyword"),
    ],
)
def test_output_registers_invalid_input_raises(
    register_name: str, output_registers: list[str] | str, error: type[Exception], match: str
) -> None:
    """Invalid output_registers raise instead of producing a wrong program."""
    circuit = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, register_name))
    circuit.measure(0, 0)

    with pytest.raises(error, match=match):
        compile_to_oq3(circuit, output_registers=output_registers)
