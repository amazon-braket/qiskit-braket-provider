"""Tests for verbatim box support in to_braket() function."""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, BoxOp
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates

from braket.circuits import Circuit
from braket.ir.openqasm import Program
from qiskit_braket_provider.providers.adapter import (
    _extract_verbatim_boxes,
    _restore_verbatim_boxes,
    to_braket,
    to_qiskit,
)

VERBATIM_LABEL = "verbatim"
NUM_QUBITS = 2
QUBIT_PAIR = [0, 1]


def _make_box_circuit(num_qubits: int, gates: list[tuple[str, list[int]]]) -> QuantumCircuit:
    """Create a QuantumCircuit with the given gates applied.

    Args:
        num_qubits: Number of qubits.
        gates: List of (gate_name, qubit_args) tuples.
    """
    qc = QuantumCircuit(num_qubits)
    for gate_name, qubits in gates:
        getattr(qc, gate_name)(*qubits)
    return qc


def _gate_info(braket_circuit: Circuit) -> list[tuple[str, list[int]]]:
    """Extract (name, target) list from a Braket circuit."""
    return [(instr.operator.name, instr.target) for instr in braket_circuit.instructions]


def _to_qiskit_input(source: str, use_program: bool) -> str | Program:
    """Return either a Program object or raw string for to_qiskit."""
    return Program(source=source) if use_program else source


@pytest.fixture
def single_box_qasm() -> str:
    """OpenQASM with one verbatim box followed by a gate."""
    return """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
    cnot $0, $1;
}
x $1;
"""


@pytest.fixture
def multi_box_qasm() -> str:
    """OpenQASM with two verbatim boxes separated by a gate."""
    return """
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


@pytest.fixture
def mixed_qasm() -> str:
    """OpenQASM with non-verbatim gate, verbatim box, then non-verbatim gate."""
    return """
OPENQASM 3.0;
x $0;
#pragma braket verbatim
box {
    h $0;
    cnot $0, $1;
}
y $1;
"""


@pytest.fixture
def h_cx_circuit() -> QuantumCircuit:
    """2-qubit circuit with H on q0 and CX on q0,q1."""
    return _make_box_circuit(NUM_QUBITS, [("h", [0]), ("cx", [0, 1])])


@pytest.fixture
def h_circuit() -> QuantumCircuit:
    """1-qubit circuit with H on q0."""
    return _make_box_circuit(NUM_QUBITS, [("h", [0])])


@pytest.fixture
def cx_circuit() -> QuantumCircuit:
    """2-qubit circuit with CX on q0,q1."""
    return _make_box_circuit(NUM_QUBITS, [("cx", [0, 1])])


@pytest.mark.parametrize(
    "inner_gates, expected_gate_names, expected_qubits",
    [
        ([("h", [0]), ("cx", [0, 1])], ["h", "cx"], QUBIT_PAIR),
        ([], [], QUBIT_PAIR),  # empty verbatim box
    ],
    ids=["single_box_with_gates", "empty_box"],
)
def test_verbatim_box_extraction(
    inner_gates: list[tuple[str, list[int]]],
    expected_gate_names: list[str],
    expected_qubits: list[int],
):
    inner = _make_box_circuit(NUM_QUBITS, inner_gates)
    main = QuantumCircuit(NUM_QUBITS)
    main.append(BoxOp(inner, label=VERBATIM_LABEL), QUBIT_PAIR)

    modified, boxes = _extract_verbatim_boxes(main, VERBATIM_LABEL)

    assert len(modified.data) == 1
    assert isinstance(modified.data[0].operation, Barrier)
    assert modified.data[0].operation.label == VERBATIM_LABEL

    assert len(boxes) == 1
    box_circuit, qubit_indices = boxes[0]
    assert [d.operation.name for d in box_circuit.data] == expected_gate_names
    assert qubit_indices == expected_qubits


def test_multiple_verbatim_boxes_extraction(h_circuit: QuantumCircuit, cx_circuit: QuantumCircuit):
    main = QuantumCircuit(NUM_QUBITS)
    main.append(BoxOp(h_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)
    main.x(1)
    main.append(BoxOp(cx_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)

    modified, boxes = _extract_verbatim_boxes(main, VERBATIM_LABEL)

    barriers = [i for i in modified.data if isinstance(i.operation, Barrier)]
    assert len(barriers) == 2
    assert all(b.operation.label == VERBATIM_LABEL for b in barriers)
    assert len([i for i in modified.data if i.operation.name == "x"]) == 1

    assert len(boxes) == 2
    assert boxes[0][0].data[0].operation.name == "h"
    assert boxes[1][0].data[0].operation.name == "cx"


def test_circuit_without_verbatim_boxes():
    main = _make_box_circuit(NUM_QUBITS, [("h", [0]), ("cx", [0, 1])])
    modified, boxes = _extract_verbatim_boxes(main, VERBATIM_LABEL)

    assert len(modified.data) == 2
    assert [d.operation.name for d in modified.data] == ["h", "cx"]
    assert len(boxes) == 0


def test_non_verbatim_boxop_not_extracted(h_circuit: QuantumCircuit):
    main = QuantumCircuit(NUM_QUBITS)
    main.append(BoxOp(h_circuit, label="other_label"), QUBIT_PAIR)

    modified, boxes = _extract_verbatim_boxes(main, VERBATIM_LABEL)

    assert len(boxes) == 0
    assert len(modified.data) == 1
    assert isinstance(modified.data[0].operation, BoxOp)
    assert modified.data[0].operation.label == "other_label"


def test_single_verbatim_box_restoration(h_cx_circuit: QuantumCircuit):
    transpiled = QuantumCircuit(NUM_QUBITS)
    transpiled.append(Barrier(NUM_QUBITS, label=VERBATIM_LABEL), QUBIT_PAIR)

    restored = _restore_verbatim_boxes(transpiled, [(h_cx_circuit, QUBIT_PAIR)], VERBATIM_LABEL)

    assert len(restored.data) == 2
    assert restored.data[0].operation.name == "h"
    assert restored.data[1].operation.name == "cx"
    assert restored.find_bit(restored.data[0].qubits[0]).index == 0
    assert restored.find_bit(restored.data[1].qubits[0]).index == 0
    assert restored.find_bit(restored.data[1].qubits[1]).index == 1


def test_multiple_verbatim_boxes_restoration(h_circuit: QuantumCircuit, cx_circuit: QuantumCircuit):
    transpiled = QuantumCircuit(NUM_QUBITS)
    transpiled.append(Barrier(NUM_QUBITS, label=VERBATIM_LABEL), QUBIT_PAIR)
    transpiled.x(1)
    transpiled.append(Barrier(NUM_QUBITS, label=VERBATIM_LABEL), QUBIT_PAIR)

    restored = _restore_verbatim_boxes(
        transpiled, [(h_circuit, QUBIT_PAIR), (cx_circuit, QUBIT_PAIR)], VERBATIM_LABEL
    )

    gate_names = [i.operation.name for i in restored.data]
    assert gate_names == ["h", "x", "cx"]


@pytest.mark.parametrize(
    "num_barriers, num_boxes, error_match",
    [
        (2, 1, "Compiler error while processing verbatim boxes.*Illegal barriers"),
        (1, 2, "Compiler error while processing verbatim boxes.*Expected"),
    ],
    ids=["too_many_barriers", "too_few_barriers"],
)
def test_barrier_box_count_mismatch(num_barriers: int, num_boxes: int, error_match: str):
    transpiled = QuantumCircuit(NUM_QUBITS)
    for _ in range(num_barriers):
        transpiled.append(Barrier(NUM_QUBITS, label=VERBATIM_LABEL), QUBIT_PAIR)

    boxes = [(_make_box_circuit(NUM_QUBITS, [("h", [0])]), QUBIT_PAIR) for _ in range(num_boxes)]

    with pytest.raises(ValueError, match=error_match):
        _restore_verbatim_boxes(transpiled, boxes, VERBATIM_LABEL)


def test_to_braket_with_single_verbatim_box(h_cx_circuit: QuantumCircuit):
    qc = QuantumCircuit(NUM_QUBITS)
    qc.x(0)
    qc.append(BoxOp(h_cx_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)
    qc.y(1)

    bc = to_braket(qc, verbatim=False)
    info = _gate_info(bc)
    names = [n for n, _ in info]

    assert bc.qubit_count == NUM_QUBITS
    for expected in ("X", "H", "CNot", "Y"):
        assert expected in names

    indices = {
        n: next(i for i, (nm, _) in enumerate(info) if nm == n) for n in ("X", "H", "CNot", "Y")
    }
    assert indices["X"] < indices["H"] < indices["CNot"] < indices["Y"]
    assert info[indices["X"]][1] == [0]
    assert info[indices["H"]][1] == [0]
    assert info[indices["CNot"]][1] == QUBIT_PAIR
    assert info[indices["Y"]][1] == [1]


def test_to_braket_with_multiple_verbatim_boxes(
    h_circuit: QuantumCircuit, cx_circuit: QuantumCircuit
):
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(BoxOp(h_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)
    qc.x(1)
    qc.append(BoxOp(cx_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)

    bc = to_braket(qc, verbatim=False)
    info = _gate_info(bc)

    assert bc.qubit_count == NUM_QUBITS
    indices = {n: next(i for i, (nm, _) in enumerate(info) if nm == n) for n in ("H", "X", "CNot")}
    assert indices["H"] < indices["X"] < indices["CNot"]
    assert info[indices["H"]][1] == [0]
    assert info[indices["X"]][1] == [1]
    assert info[indices["CNot"]][1] == QUBIT_PAIR


def test_to_braket_with_custom_verbatim_box_name(h_cx_circuit: QuantumCircuit):
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(BoxOp(h_cx_circuit, label="custom_verbatim"), QUBIT_PAIR)

    bc = to_braket(qc, verbatim=False, verbatim_box_name="custom_verbatim")
    info = _gate_info(bc)
    names = [n for n, _ in info]

    assert bc.qubit_count == NUM_QUBITS
    assert "H" in names
    assert "CNot" in names
    h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
    cnot_idx = next(i for i, (n, _) in enumerate(info) if n == "CNot")
    assert info[h_idx][1] == [0]
    assert info[cnot_idx][1] == QUBIT_PAIR
    assert h_idx < cnot_idx


def test_to_braket_backward_compatibility():
    qc = _make_box_circuit(NUM_QUBITS, [("h", [0]), ("cx", [0, 1])])
    bc = to_braket(qc, verbatim=False)
    info = _gate_info(bc)
    names = [n for n, _ in info]

    assert bc.qubit_count == NUM_QUBITS
    assert "H" in names
    assert "CNot" in names
    h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
    cnot_idx = next(i for i, (n, _) in enumerate(info) if n == "CNot")
    assert info[h_idx][1] == [0]
    assert info[cnot_idx][1] == QUBIT_PAIR
    assert h_idx < cnot_idx


@pytest.mark.parametrize(
    "verbatim, layout_method",
    [
        (True, None),
        (False, None),
        (False, "dense"),
    ],
    ids=["verbatim_true", "trivial_layout", "layout_override"],
)
def test_to_braket_verbatim_and_layout_options(
    verbatim: bool, layout_method: str | None, h_cx_circuit: QuantumCircuit
):
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(BoxOp(h_cx_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)

    bc = to_braket(qc, verbatim=verbatim, layout_method=layout_method)
    info = _gate_info(bc)
    names = [n for n, _ in info]

    assert bc.qubit_count == NUM_QUBITS
    assert "H" in names
    assert "CNot" in names
    h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
    cnot_idx = next(i for i, (n, _) in enumerate(info) if n == "CNot")
    assert info[h_idx][1] == [0]
    assert info[cnot_idx][1] == QUBIT_PAIR
    assert h_idx < cnot_idx


def test_to_braket_raises_on_pass_manager_with_verbatim_boxes(h_circuit: QuantumCircuit):
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(BoxOp(h_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)

    with pytest.raises(
        ValueError, match="Custom pass_manager is not supported with verbatim boxes"
    ):
        to_braket(qc, verbatim=False, pass_manager=PassManager([Optimize1qGates()]))


def test_to_braket_raises_on_barrier_labeled_as_verbatim_box():
    qc = QuantumCircuit(NUM_QUBITS)
    qc.x(0)
    qc.append(Barrier(NUM_QUBITS, label=VERBATIM_LABEL), QUBIT_PAIR)
    qc.y(1)

    with pytest.raises(
        ValueError,
        match="Cannot have a Barrier labeled with the same label used for verbatim boxes",
    ):
        to_braket(qc, verbatim=False)


def test_to_braket_with_multiple_circuits_with_verbatim_boxes(h_cx_circuit: QuantumCircuit):
    qc1 = QuantumCircuit(NUM_QUBITS)
    qc1.x(0)
    qc1.append(BoxOp(h_cx_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)

    inner2 = _make_box_circuit(3, [("h", [0]), ("h", [1]), ("ccx", [0, 1, 2])])
    qc2 = QuantumCircuit(3)
    qc2.append(BoxOp(inner2, label=VERBATIM_LABEL), [0, 1, 2])
    qc2.z(2)

    qc3 = _make_box_circuit(NUM_QUBITS, [("h", [0]), ("cx", [0, 1])])

    results = to_braket([qc1, qc2, qc3], verbatim=False)

    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0].qubit_count == NUM_QUBITS
    assert results[1].qubit_count == 3
    assert results[2].qubit_count == NUM_QUBITS


@pytest.mark.parametrize("use_program", [True, False], ids=["braket_program", "openqasm"])
def test_round_trip_single_verbatim_box(use_program: bool, single_box_qasm: str):
    qc = to_qiskit(_to_qiskit_input(single_box_qasm, use_program))

    box_ops = [
        i for i in qc.data if hasattr(i.operation, "label") and i.operation.label == VERBATIM_LABEL
    ]
    assert len(box_ops) == 1

    bc = to_braket(qc, verbatim=False)
    info = _gate_info(bc)
    names = [n for n, _ in info]

    assert bc.qubit_count == NUM_QUBITS
    for expected in ("H", "CNot", "X"):
        assert expected in names

    indices = {n: next(i for i, (nm, _) in enumerate(info) if nm == n) for n in ("H", "CNot", "X")}
    assert indices["H"] < indices["X"]
    assert indices["CNot"] < indices["X"]
    assert info[indices["H"]][1] == [0]
    assert info[indices["CNot"]][1] == QUBIT_PAIR
    assert info[indices["X"]][1] == [1]


@pytest.mark.parametrize("use_program", [True, False], ids=["braket_program", "openqasm"])
def test_round_trip_multiple_verbatim_boxes(use_program: bool, multi_box_qasm: str):
    qc = to_qiskit(_to_qiskit_input(multi_box_qasm, use_program))

    box_ops = [
        i for i in qc.data if hasattr(i.operation, "label") and i.operation.label == VERBATIM_LABEL
    ]
    assert len(box_ops) == 2

    bc = to_braket(qc, verbatim=False)
    info = _gate_info(bc)

    assert bc.qubit_count == NUM_QUBITS
    indices = {n: next(i for i, (nm, _) in enumerate(info) if nm == n) for n in ("H", "X", "CNot")}
    assert indices["H"] < indices["X"] < indices["CNot"]
    assert info[indices["H"]][1] == [0]
    assert info[indices["X"]][1] == [1]
    assert info[indices["CNot"]][1] == QUBIT_PAIR


@pytest.mark.parametrize("use_program", [True, False], ids=["braket_program", "openqasm"])
def test_round_trip_custom_verbatim_box_name(use_program: bool):
    qasm = """
OPENQASM 3.0;
#pragma braket verbatim
box {
    h $0;
    cnot $2, $4;
}
"""
    label = "custom_verbatim" if use_program else "my_custom_verbatim"
    qc = to_qiskit(_to_qiskit_input(qasm, use_program), verbatim_box_name=label)

    box_ops = [i for i in qc.data if hasattr(i.operation, "label") and i.operation.label == label]
    assert len(box_ops) == 1

    bc = to_braket(qc, verbatim=False, verbatim_box_name=label)
    info = _gate_info(bc)
    names = [n for n, _ in info]

    assert bc.qubit_count == 3
    assert "H" in names
    assert "CNot" in names
    h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
    cnot_idx = next(i for i, (n, _) in enumerate(info) if n == "CNot")
    assert info[h_idx][1] == [0]
    assert info[cnot_idx][1] == [2, 4]
    assert h_idx < cnot_idx


@pytest.mark.parametrize("use_program", [True, False], ids=["braket_program", "openqasm"])
def test_round_trip_mixed_verbatim_and_non_verbatim(use_program: bool, mixed_qasm: str):
    qc = to_qiskit(_to_qiskit_input(mixed_qasm, use_program))
    bc = to_braket(qc, verbatim=False)
    info = _gate_info(bc)
    names = [n for n, _ in info]

    assert bc.qubit_count == NUM_QUBITS
    assert "H" in names
    assert "CNot" in names

    h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
    cnot_idx = next(i for i, (n, _) in enumerate(info) if n == "CNot")
    assert info[h_idx][1] == [0]
    assert info[cnot_idx][1] == QUBIT_PAIR
    assert h_idx < cnot_idx


def test_round_trip_multiple_verbatim_boxes_openqasm_3_qubits():
    """OpenQASM-specific test with 3 qubits and 2 CNot gates."""
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
    cnot $1, $2;
}
"""
    qc = to_qiskit(qasm)
    box_ops = [
        i for i in qc.data if hasattr(i.operation, "label") and i.operation.label == VERBATIM_LABEL
    ]
    assert len(box_ops) == 2

    bc = to_braket(qc, verbatim=False)
    info = _gate_info(bc)

    assert bc.qubit_count == 3
    h_idx = next(i for i, (n, _) in enumerate(info) if n == "H")
    x_idx = next(i for i, (n, _) in enumerate(info) if n == "X")
    cnot_indices = [i for i, (n, _) in enumerate(info) if n == "CNot"]

    assert h_idx < x_idx
    assert len(cnot_indices) == 2
    assert all(ci > x_idx for ci in cnot_indices)
    assert info[h_idx][1] == [0]
    assert info[x_idx][1] == [1]
    assert info[cnot_indices[0]][1] == QUBIT_PAIR
    assert info[cnot_indices[1]][1] == [1, 2]
