"""Tests for verbatim box support in to_braket() function."""

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, BoxOp, Parameter
from qiskit.circuit.library import CXGate, CZGate, HGate, Measure, RXGate, XGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import InstructionProperties, PassManager, Target
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import Optimize1qGates

from braket.circuits import Circuit
from braket.ir.openqasm import Program
from qiskit_braket_provider.providers.adapter import (
    to_braket,
    to_qiskit,
)
from qiskit_braket_provider.providers.compilation import _compile
from qiskit_braket_provider.providers.passes import (
    ExtractVerbatimBoxes,
    RestoreVerbatimBoxes,
)
from qiskit_braket_provider.providers.passes.verbatim_passes import (
    VerbatimPlaceholder,
    _indexed_label,
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

    pm = PassManager([ExtractVerbatimBoxes(VERBATIM_LABEL)])
    modified = pm.run(main)

    assert len(modified.data) == 1
    assert isinstance(modified.data[0].operation, VerbatimPlaceholder)
    assert modified.data[0].operation.label.startswith(VERBATIM_LABEL)

    verbatim_boxes = pm.property_set["verbatim_boxes"]
    assert len(verbatim_boxes) == 1
    box_circuit = next(iter(verbatim_boxes.values()))
    assert [d.operation.name for d in box_circuit.data] == expected_gate_names

    barrier_qubits = [modified.find_bit(q).index for q in modified.data[0].qubits]
    assert barrier_qubits == expected_qubits


def test_multiple_verbatim_boxes_extraction(h_circuit: QuantumCircuit, cx_circuit: QuantumCircuit):
    main = QuantumCircuit(NUM_QUBITS)
    main.append(BoxOp(h_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)
    main.x(1)
    main.append(BoxOp(cx_circuit, label=VERBATIM_LABEL), QUBIT_PAIR)

    pm = PassManager([ExtractVerbatimBoxes(VERBATIM_LABEL)])
    modified = pm.run(main)

    barriers = [i for i in modified.data if isinstance(i.operation, VerbatimPlaceholder)]
    assert len(barriers) == 2
    assert all(b.operation.label.startswith(VERBATIM_LABEL) for b in barriers)
    assert len([i for i in modified.data if i.operation.name == "x"]) == 1

    for barrier_instr in barriers:
        barrier_qubits = [modified.find_bit(q).index for q in barrier_instr.qubits]
        assert barrier_qubits == QUBIT_PAIR

    verbatim_boxes = pm.property_set["verbatim_boxes"]
    assert len(verbatim_boxes) == 2
    box_circuits = list(verbatim_boxes.values())
    assert box_circuits[0].data[0].operation.name == "h"
    assert box_circuits[1].data[0].operation.name == "cx"


def test_circuit_without_verbatim_boxes():
    main = _make_box_circuit(NUM_QUBITS, [("h", [0]), ("cx", [0, 1])])
    pm = PassManager([ExtractVerbatimBoxes(VERBATIM_LABEL)])
    modified = pm.run(main)

    assert len(modified.data) == 2
    assert [d.operation.name for d in modified.data] == ["h", "cx"]
    assert pm.property_set["verbatim_boxes"] == {}


def test_non_verbatim_boxop_not_extracted(h_circuit: QuantumCircuit):
    main = QuantumCircuit(NUM_QUBITS)
    main.append(BoxOp(h_circuit, label="other_label"), QUBIT_PAIR)

    pm = PassManager([ExtractVerbatimBoxes(VERBATIM_LABEL)])
    modified = pm.run(main)

    assert pm.property_set["verbatim_boxes"] == {}
    assert len(modified.data) == 1
    assert isinstance(modified.data[0].operation, BoxOp)
    assert modified.data[0].operation.label == "other_label"


def test_single_verbatim_box_restoration(h_cx_circuit: QuantumCircuit):
    """RestoreVerbatimBoxes replaces a labeled placeholder with stashed box contents."""
    label = _indexed_label(VERBATIM_LABEL, 0)
    transpiled = QuantumCircuit(NUM_QUBITS)
    transpiled.append(VerbatimPlaceholder(NUM_QUBITS, 0, label=label), QUBIT_PAIR)

    restore_pass = RestoreVerbatimBoxes(VERBATIM_LABEL)
    restore_pass.property_set["verbatim_boxes"] = {label: h_cx_circuit}

    dag = circuit_to_dag(transpiled)
    restored = dag_to_circuit(restore_pass.run(dag))

    assert len(restored.data) == 2
    assert restored.data[0].operation.name == "h"
    assert restored.data[1].operation.name == "cx"
    assert restored.find_bit(restored.data[0].qubits[0]).index == 0
    assert restored.find_bit(restored.data[1].qubits[0]).index == 0
    assert restored.find_bit(restored.data[1].qubits[1]).index == 1


def test_multiple_verbatim_boxes_restoration(h_circuit: QuantumCircuit, cx_circuit: QuantumCircuit):
    """RestoreVerbatimBoxes replaces multiple labeled placeholders with stashed box contents."""
    label_0 = _indexed_label(VERBATIM_LABEL, 0)
    label_1 = _indexed_label(VERBATIM_LABEL, 1)
    transpiled = QuantumCircuit(NUM_QUBITS)
    transpiled.append(VerbatimPlaceholder(NUM_QUBITS, 0, label=label_0), QUBIT_PAIR)
    transpiled.x(1)
    transpiled.append(VerbatimPlaceholder(NUM_QUBITS, 0, label=label_1), QUBIT_PAIR)

    restore_pass = RestoreVerbatimBoxes(VERBATIM_LABEL)
    restore_pass.property_set["verbatim_boxes"] = {
        label_0: h_circuit,
        label_1: cx_circuit,
    }

    dag = circuit_to_dag(transpiled)
    restored = dag_to_circuit(restore_pass.run(dag))

    gate_names = [i.operation.name for i in restored.data]
    assert gate_names == ["h", "x", "cx"]

    assert restored.find_bit(restored.data[0].qubits[0]).index == 0  # h on q0
    assert restored.find_bit(restored.data[1].qubits[0]).index == 1  # x on q1
    assert restored.find_bit(restored.data[2].qubits[0]).index == 0  # cx control q0
    assert restored.find_bit(restored.data[2].qubits[1]).index == 1  # cx target q1


def test_extract_raises_on_barrier_labeled_verbatim():
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(Barrier(NUM_QUBITS, label=VERBATIM_LABEL), QUBIT_PAIR)

    pm = PassManager([ExtractVerbatimBoxes(VERBATIM_LABEL)])
    with pytest.raises(ValueError, match="conflicts with the verbatim box label"):
        pm.run(qc)


def test_extract_raises_on_conflicting_barrier_label():
    """Test that ExtractVerbatimBoxes raises on conflicting barrier labels."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(Barrier(NUM_QUBITS, label="verbatim__0"), QUBIT_PAIR)

    pm = PassManager([ExtractVerbatimBoxes(VERBATIM_LABEL)])
    with pytest.raises(ValueError, match="conflicts with the verbatim box label"):
        pm.run(qc)


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


def test_to_braket_handles_verbatim_box_with_clbits():
    """to_braket on a verbatim BoxOp carrying clbits succeeds."""
    body = QuantumCircuit(1, 1)
    body.h(0)
    body.measure(0, 0)

    qc = QuantumCircuit(1, 1)
    qc.append(BoxOp(body, label=VERBATIM_LABEL), qc.qubits, qc.clbits)

    bc = to_braket(qc, verbatim=True)

    source = bc.to_ir(ir_type="OPENQASM").source
    expected = (
        "OPENQASM 3.0;\nbit[1] b;\n#pragma braket verbatim\nbox{\nh $0;\n}\nb[0] = measure $0;"
    )
    assert source == expected


def test_to_braket_verbatim_box_with_clbits_on_subset_of_qubits():
    """Verbatim BoxOp on a qubit subset of a larger circuit with measurements."""
    body = QuantumCircuit(2, 2)
    body.x(0)
    body.measure(0, 0)
    body.measure(1, 1)

    qc = QuantumCircuit(3, 2)
    qc.append(BoxOp(body, label=VERBATIM_LABEL), [qc.qubits[0], qc.qubits[1]], qc.clbits)

    bc = to_braket(qc, verbatim=True)

    source = bc.to_ir(ir_type="OPENQASM").source
    expected = (
        "OPENQASM 3.0;\nbit[2] b;\n#pragma braket verbatim\nbox{\nx $0;\n}\n"
        "b[0] = measure $0;\nb[1] = measure $1;"
    )
    assert source == expected


def test_verbatim_box_with_clbits_through_transpilation():
    """Verbatim BoxOp with clbits goes through the pass pipeline when transpiling."""
    body = QuantumCircuit(1, 1)
    body.h(0)
    body.measure(0, 0)

    qc = QuantumCircuit(2, 1)
    qc.x(0)
    qc.append(BoxOp(body, label=VERBATIM_LABEL), [qc.qubits[1]], qc.clbits)

    bc = to_braket(qc, basis_gates=["x", "h", "cx", "measure"])

    source = bc.to_ir(ir_type="OPENQASM").source
    expected = "OPENQASM 3.0;\nbit[1] b;\nqubit[2] q;\nx q[0];\nh q[1];\nb[0] = measure q[1];"
    assert source == expected


def test_verbatim_box_with_clbits_on_subset_through_transpilation():
    """Verbatim BoxOp with clbits on qubit subset goes through pass pipeline."""
    body = QuantumCircuit(2, 2)
    body.x(0)
    body.measure(0, 0)
    body.measure(1, 1)

    qc = QuantumCircuit(3, 2)
    qc.append(BoxOp(body, label=VERBATIM_LABEL), [qc.qubits[0], qc.qubits[1]], qc.clbits)

    bc = to_braket(qc, basis_gates=["x", "h", "cx", "measure"])

    source = bc.to_ir(ir_type="OPENQASM").source
    expected = (
        "OPENQASM 3.0;\nbit[2] b;\nqubit[2] q;\nx q[0];\nb[0] = measure q[0];\nb[1] = measure q[1];"
    )
    assert source == expected


def test_to_braket_round_trip_preserves_verbatim_box_with_measurement():
    """QASM → Qiskit → Braket QASM preserves verbatim gates and non-identity clbit mapping."""
    src = """
    OPENQASM 3.0;
    bit[2] b;
    #pragma braket verbatim
    box {
        h $0;
        cnot $0, $1;
        b[1] = measure $0;
        b[0] = measure $1;
    }
    """
    qc = to_qiskit(src)
    bc = to_braket(qc, verbatim=True)
    out = bc.to_ir(ir_type="OPENQASM").source
    expected = (
        "OPENQASM 3.0;\nbit[2] b;\n#pragma braket verbatim\nbox{\n"
        "h $0;\ncnot $0, $1;\n}\nb[0] = measure $1;\nb[1] = measure $0;"
    )
    assert out == expected


def test_verbatim_boxes_without_transpilation_needed():
    """Cover the elif path: verbatim boxes present but no transpilation triggers."""

    body = QuantumCircuit(NUM_QUBITS)
    body.h(0)

    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(BoxOp(body, label=VERBATIM_LABEL), QUBIT_PAIR)
    result = _compile(qc, basis_gates={"h", "cx", "box"})
    assert result.circuits[0].data[0].operation.name == "h"


def test_restore_without_extract_is_noop():
    """RestoreVerbatimBoxes with empty property_set returns dag unchanged."""
    qc = QuantumCircuit(NUM_QUBITS)
    qc.h(0)

    pm = PassManager([RestoreVerbatimBoxes(VERBATIM_LABEL)])
    result = pm.run(qc)
    assert len(result.data) == 1
    assert result.data[0].operation.name == "h"
    assert [result.find_bit(q).index for q in result.data[0].qubits] == [0]


def test_restore_raises_on_mismatched_barrier_label():
    """RestoreVerbatimBoxes raises when a verbatim placeholder has no matching box."""

    qc = QuantumCircuit(NUM_QUBITS)
    label = _indexed_label(VERBATIM_LABEL, 0)
    qc.append(VerbatimPlaceholder(NUM_QUBITS, 0, label=label), QUBIT_PAIR)

    restore_pass = RestoreVerbatimBoxes(VERBATIM_LABEL)
    # Set verbatim_boxes with a different label than what's in the circuit
    different_label = _indexed_label(VERBATIM_LABEL, 99)
    body = QuantumCircuit(NUM_QUBITS)
    body.h(0)
    restore_pass.property_set["verbatim_boxes"] = {different_label: body}

    dag = circuit_to_dag(qc)
    with pytest.raises(RuntimeError, match="has no matching box"):
        restore_pass.run(dag)


def test_restore_raises_on_leftover_boxes():
    """RestoreVerbatimBoxes raises when boxes remain after processing."""

    qc = QuantumCircuit(NUM_QUBITS)
    qc.h(0)

    restore_pass = RestoreVerbatimBoxes(VERBATIM_LABEL)
    # Set verbatim_boxes that will never be consumed
    label = _indexed_label(VERBATIM_LABEL, 0)
    body = QuantumCircuit(NUM_QUBITS)
    body.h(0)
    restore_pass.property_set["verbatim_boxes"] = {label: body}

    dag = circuit_to_dag(qc)
    with pytest.raises(RuntimeError, match="stashed verbatim boxes were not restored"):
        restore_pass.run(dag)


def test_extract_restore_with_non_verbatim_barrier():
    """A non-verbatim barrier is preserved through extract/restore."""
    body = QuantumCircuit(NUM_QUBITS)
    body.h(0)

    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(Barrier(NUM_QUBITS), QUBIT_PAIR)
    qc.append(BoxOp(body, label=VERBATIM_LABEL), QUBIT_PAIR)

    pm = PassManager([ExtractVerbatimBoxes(VERBATIM_LABEL), RestoreVerbatimBoxes(VERBATIM_LABEL)])
    result = pm.run(qc)

    # The unlabeled barrier is preserved, and the verbatim box is unpacked
    assert result.data[0].operation.name == "barrier"
    assert result.data[0].operation.label is None
    assert result.data[1].operation.name == "h"


def test_verbatim_with_target_runs_contains_instruction():
    """ContainsInstruction pass runs when verbatim boxes are present with a target."""
    target = Target(num_qubits=2)
    target.add_instruction(HGate(), {(i,): None for i in range(2)})
    target.add_instruction(CXGate(), {(0, 1): None})
    target.add_instruction(Measure(), {(i,): None for i in range(2)})

    body = QuantumCircuit(2)
    body.h(0)
    body.cx(0, 1)

    qc = QuantumCircuit(2)
    qc.append(BoxOp(body, label=VERBATIM_LABEL), [0, 1])

    pass_names = []

    def cb(**kwargs):
        p = kwargs.get("pass_")
        if p:
            pass_names.append(type(p).__name__)

    _compile(qc, target=target, callback=cb)
    assert "ContainsInstruction" in pass_names


def test_verbatim_with_if_else_on_unsupported_backend_raises_clean_error():
    """Verbatim box + if_else on a backend without control-flow gives a clear error."""
    target = Target(num_qubits=3)
    target.add_instruction(HGate(), {(i,): None for i in range(3)})
    target.add_instruction(XGate(), {(i,): None for i in range(3)})
    target.add_instruction(CXGate(), {(0, 1): None, (1, 2): None})
    target.add_instruction(Measure(), {(i,): None for i in range(3)})

    qc = to_qiskit("""
    OPENQASM 3.0;
    bit[1] c;
    #pragma braket verbatim
    box {
        h $0;
        cnot $0, $1;
    }
    c[0] = measure $0;
    if (c[0] == 1) {
        x $2;
    }
    """)

    with pytest.raises(TranspilerError, match="control-flow"):
        to_braket(qc, target=target)


def test_verbatim_box_preserves_reversed_qubit_order():
    """Verbatim box on reversed qubits preserves outer qubit indices."""
    body = QuantumCircuit(NUM_QUBITS)
    body.cx(0, 1)

    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(BoxOp(body, label=VERBATIM_LABEL), [qc.qubits[1], qc.qubits[0]])

    out = _compile(qc, basis_gates={"cx"}).circuits[0]
    cx = next(i for i in out.data if i.operation.name == "cx")
    assert [out.find_bit(q).index for q in cx.qubits] == [1, 0]


def test_verbatim_box_preserves_non_contiguous_qubit_order():
    """Verbatim box on non-contiguous qubits preserves outer qubit indices."""
    body = QuantumCircuit(2)
    body.cx(0, 1)

    qc = QuantumCircuit(3)
    qc.append(BoxOp(body, label=VERBATIM_LABEL), [qc.qubits[2], qc.qubits[0]])

    out = _compile(qc, basis_gates={"cx"}).circuits[0]
    cx = next(i for i in out.data if i.operation.name == "cx")
    assert [out.find_bit(q).index for q in cx.qubits] == [2, 0]


def _build_target_2q() -> Target:
    """Helper to build a minimal 2-qubit target with rx/h/cz/measure."""
    target = Target(num_qubits=2)
    theta = Parameter("theta")
    target.add_instruction(RXGate(theta), {(i,): InstructionProperties() for i in range(2)})
    target.add_instruction(HGate(), {(i,): InstructionProperties() for i in range(2)})
    target.add_instruction(
        CZGate(), {(0, 1): InstructionProperties(), (1, 0): InstructionProperties()}
    )
    target.add_instruction(Measure(), {(i,): InstructionProperties() for i in range(2)})
    return target


@pytest.mark.parametrize(
    "compile_kwargs",
    [
        pytest.param(
            {"target": _build_target_2q(), "optimization_level": 3, "seed_transpiler": 42},
            id="target_opt3",
        ),
        pytest.param(
            {
                "basis_gates": ["rx", "cz", "h", "measure"],
                "optimization_level": 3,
                "seed_transpiler": 42,
            },
            id="basis_gates_opt3",
        ),
    ],
)
def test_verbatim_box_with_clbits_no_panic(compile_kwargs: dict):
    """OQ3 verbatim box with classical registers does not panic during compilation."""
    source = """\
OPENQASM 3.0;
bit[2] c;
qubit[2] q;
h q[0];
#pragma braket verbatim
box { rx(0.5) q[0]; cz q[0], q[1]; }
c[0] = measure q[0];
c[1] = measure q[1];
"""
    qc = to_qiskit(source)
    result = _compile(qc, **compile_kwargs)
    circ = result.circuits[0]
    ops = [instr.operation.name for instr in circ.data]
    assert "rx" in ops
    assert "cz" in ops
    assert ops.count("measure") == 2


@pytest.mark.parametrize(
    "compile_kwargs",
    [
        pytest.param(
            {"target": _build_target_2q(), "optimization_level": 3, "seed_transpiler": 42},
            id="target_opt3",
        ),
        pytest.param(
            {
                "basis_gates": ["rx", "cz", "h", "measure"],
                "optimization_level": 3,
                "seed_transpiler": 42,
            },
            id="basis_gates_opt3",
        ),
    ],
)
def test_verbatim_box_with_mcm_preserves_order(compile_kwargs: dict):
    """OQ3 verbatim box with mid-circuit measurement preserves operation order."""
    source = """\
OPENQASM 3.0;
bit[2] c;
qubit[2] q;
h q[0];
#pragma braket verbatim
box { rx(0.5) q[0]; c[0] = measure q[0]; h q[0]; cz q[0], q[1]; }
c[1] = measure q[1];
"""
    qc = to_qiskit(source)
    result = _compile(qc, **compile_kwargs)
    circ = result.circuits[0]
    ops = [instr.operation.name for instr in circ.data]
    assert ops == ["h", "rx", "measure", "h", "cz", "measure"]


def test_restore_skips_placeholder_with_non_matching_label():
    """RestoreVerbatimBoxes ignores placeholders whose label doesn't match the verbatim pattern."""
    label = _indexed_label(VERBATIM_LABEL, 0)
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(VerbatimPlaceholder(NUM_QUBITS, 0, label="unrelated_label"), QUBIT_PAIR)
    qc.append(VerbatimPlaceholder(NUM_QUBITS, 0, label=label), QUBIT_PAIR)

    body = QuantumCircuit(NUM_QUBITS)
    body.h(0)

    restore_pass = RestoreVerbatimBoxes(VERBATIM_LABEL)
    restore_pass.property_set["verbatim_boxes"] = {label: body}

    dag = circuit_to_dag(qc)
    restored = dag_to_circuit(restore_pass.run(dag))

    ops = [(instr.operation.name, getattr(instr.operation, "label", None)) for instr in restored.data]
    assert ("barrier", "unrelated_label") in ops
    assert ("h", None) in ops
