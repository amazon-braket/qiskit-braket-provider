"""Tests for the physical_qubit_labels translation in to_qiskit.

Covers _QiskitProgramContext.get_qubits() with physical_qubit_labels supplied:
$N references are translated through the label map, unknown labels raise
ValueError, declared registers and Circuit inputs are unaffected.
"""

from collections.abc import Callable

import pytest

from braket.circuits import Circuit
from braket.ir.openqasm import Program
from qiskit_braket_provider import to_qiskit

# IQM Garnet-shaped labels: 20 qubits, 1..20 (1-based, contiguous).
GARNET_LABELS = tuple(range(1, 21))

# Rigetti Cepheus-shaped labels: 108 qubit slots with qubit 8 missing,
# so 107 labels with a hole in the middle of the range.
CEPHEUS_LABELS = tuple(sorted(set(range(108)) - {8}))


@pytest.mark.parametrize(
    "op_line, labels, expected_num_qubits, expected_index",
    [
        # Bottom and top of a 1-based label range (IQM Garnet shape).
        ("x $1;", GARNET_LABELS, 1, 0),
        ("x $20;", GARNET_LABELS, 20, 19),
        # Example of a hole in the label space (Rigetti Cepheus shape).
        ("x $9;", CEPHEUS_LABELS, 9, 8),
        # Top of range on a device with a hole in the middle.
        ("x $107;", CEPHEUS_LABELS, 107, 106),
        # A $N measurement-only reference is translated the same way.
        ("bit[1] b;\nb[0] = measure $20;", GARNET_LABELS, 20, 19),
    ],
    ids=[
        "garnet_$1_bottom",
        "garnet_$20_top",
        "cepheus_$9_after_hole",
        "cepheus_$107_top",
        "garnet_measure_$20",
    ],
)
def test_physical_qubit_reference_translated(
    op_line: str, labels: tuple[int, ...], expected_num_qubits: int, expected_index: int
):
    qc = to_qiskit(
        Program(source=f"OPENQASM 3.0;\n{op_line}"),
        physical_qubit_labels=labels,
    )
    assert qc.num_qubits == expected_num_qubits
    assert qc.find_bit(qc.data[0].qubits[0]).index == expected_index


def test_two_qubit_gate_across_range():
    qc = to_qiskit(
        Program(source="OPENQASM 3.0;\ncz $1, $20;"),
        physical_qubit_labels=GARNET_LABELS,
    )
    assert qc.num_qubits == 20
    cz_qubits = qc.data[0].qubits
    assert tuple(qc.find_bit(q).index for q in cz_qubits) == (0, 19)


@pytest.mark.parametrize(
    "bad_label, labels",
    [
        (8, CEPHEUS_LABELS),  # Hole in the middle of the label space.
        (0, GARNET_LABELS),  # Below a 1-based range.
        (21, GARNET_LABELS),  # Above the range.
    ],
    ids=["cepheus_hole", "garnet_below_range", "garnet_above_range"],
)
def test_out_of_range_label_raises(bad_label: int, labels: tuple[int, ...]):
    with pytest.raises(ValueError, match=rf"\${bad_label} is not on"):
        to_qiskit(
            Program(source=f"OPENQASM 3.0;\nx ${bad_label};"),
            physical_qubit_labels=labels,
        )


def test_declared_register_not_translated():
    qc = to_qiskit(
        Program(source="OPENQASM 3.0;\nqubit[3] q;\nx q[0];"),
        physical_qubit_labels=GARNET_LABELS,
    )
    assert qc.num_qubits == 3
    assert len(qc.qregs) == 1 and qc.qregs[0].size == 3
    assert qc.find_bit(qc.data[0].qubits[0]).index == 0


def test_physical_qubit_inside_verbatim_box_translated():
    qc = to_qiskit(
        Program(source=("OPENQASM 3.0;\n#pragma braket verbatim\nbox { x $20; }\n")),
        physical_qubit_labels=GARNET_LABELS,
    )
    assert qc.num_qubits == 20
    box_op = qc.data[0].operation
    assert box_op.name == "box"
    body = box_op.body
    assert body.find_bit(body.data[0].qubits[0]).index == 19


@pytest.mark.parametrize("labels", [None, ()], ids=["none", "empty_tuple"])
def test_no_labels_or_empty_labels_preserves_legacy_behavior(labels: tuple[int, ...] | None):
    qc = to_qiskit(
        Program(source="OPENQASM 3.0;\nx $20;"),
        physical_qubit_labels=labels,
    )
    assert qc.num_qubits == 21
    assert qc.find_bit(qc.data[0].qubits[0]).index == 20


@pytest.mark.parametrize(
    "make_input",
    [
        pytest.param(lambda src: Program(source=src), id="Program"),
        pytest.param(lambda src: src, id="str"),
    ],
)
def test_program_and_str_sources_both_translated(make_input: Callable[[str], Program | str]):
    source = "OPENQASM 3.0;\nx $20;"
    qc = to_qiskit(make_input(source), physical_qubit_labels=GARNET_LABELS)
    assert qc.num_qubits == 20
    assert qc.find_bit(qc.data[0].qubits[0]).index == 19


def test_circuit_input_ignores_labels():
    circuit = Circuit().x(0)
    with_labels = to_qiskit(circuit, physical_qubit_labels=GARNET_LABELS)
    without = to_qiskit(circuit)
    assert with_labels.num_qubits == without.num_qubits
    assert [i.operation.name for i in with_labels.data] == [i.operation.name for i in without.data]
