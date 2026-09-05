"""Tests for the Braket-formatting transpiler passes."""

from collections.abc import Callable

import pytest
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import BoxOp, ClassicalRegister, Clbit, IfElseOp, QuantumRegister
from qiskit.circuit.library import CXGate, GlobalPhaseGate, PhaseGate, RGate, RXXGate
from qiskit.transpiler import PassManager

from qiskit_braket_provider.providers.braket_annotations import (
    _BRAKET_ANNOTATION_HANDLERS,
    BraketVerbatim,
    BraketVerbatimBox,
    BraketVerbatimSerializer,
)
from qiskit_braket_provider.providers.gate_mappings import (
    _BRAKET_VERBATIM_BOX_NAME,
    _QISKIT_TO_BRAKET_SHIM,
)
from qiskit_braket_provider.providers.passes import (
    ConsolidateClbits,
    MoveMeasurementsToEnd,
    RenameGates,
    WrapInVerbatimBox,
)
from qiskit_braket_provider.providers.passes.braket_formatting_passes import (
    _normalize_formatting,
    _remap_qubits,
    _shim_class_for,
)


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _two_cregs_circuit() -> QuantumCircuit:
    qr = QuantumRegister(2, "q")
    cr1 = ClassicalRegister(1, "a")
    cr2 = ClassicalRegister(1, "b")
    qc = QuantumCircuit(qr, cr1, cr2)
    qc.h(0)
    qc.measure(0, cr1[0])
    qc.measure(1, cr2[0])
    return qc


def _default_creg_circuit() -> QuantumCircuit:
    """QuantumCircuit(2, 2) creates a default register named 'c'."""
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])
    return qc


def _no_clbits_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


def _output_register_circuit() -> QuantumCircuit:
    """A circuit whose register is marked as an OpenQASM output variable."""
    creg = ClassicalRegister(2, "c")
    qc = QuantumCircuit(2)
    qc.add_register(creg)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, creg[0])
    qc.measure(1, creg[1])
    qc.metadata = {"braket_output_variables": {"c": None}}
    return qc


def _output_and_scratch_circuit() -> QuantumCircuit:
    """Output register + a plain scratch register + a loose Clbit."""
    first = ClassicalRegister(2, "first")
    scratch = ClassicalRegister(1, "scratch")
    qc = QuantumCircuit(3)
    qc.add_register(first)
    qc.add_register(scratch)
    qc.add_bits([Clbit()])
    qc.measure(0, first[0])
    qc.measure(1, first[1])
    qc.measure(2, scratch[0])
    qc.metadata = {"braket_output_variables": {"first": None}}
    return qc


def _shadow_b_circuit() -> QuantumCircuit:
    """Output register named 'b' forces plain-bit register to fall back to 'b0'."""
    b = ClassicalRegister(1, "b")
    qc = QuantumCircuit(2)
    qc.add_register(b)
    qc.add_bits([Clbit()])
    qc.measure(0, b[0])
    qc.measure(1, qc.clbits[1])
    qc.metadata = {"braket_output_variables": {"b": None}}
    return qc


def _shadow_b_and_b0_circuit() -> QuantumCircuit:
    """Output registers 'b' and 'b0' both shadow the plain-bit fallback names."""
    b = ClassicalRegister(1, "b")
    b0 = ClassicalRegister(1, "b0")
    qc = QuantumCircuit(3)
    qc.add_register(b)
    qc.add_register(b0)
    qc.add_bits([Clbit()])
    qc.measure(0, b[0])
    qc.measure(1, b0[0])
    qc.measure(2, qc.clbits[2])
    qc.metadata = {"braket_output_variables": {"b": None, "b0": None}}
    return qc


def _all_output_registers_circuit() -> QuantumCircuit:
    """Every clbit lives in an output-declared register; ``plain`` is empty."""
    a = ClassicalRegister(1, "a")
    b = ClassicalRegister(1, "b")
    qc = QuantumCircuit(2)
    qc.add_register(a)
    qc.add_register(b)
    qc.measure(0, a[0])
    qc.measure(1, b[0])
    qc.metadata = {"braket_output_variables": {"a": None, "b": None}}
    return qc


def _mid_measure_circuit() -> QuantumCircuit:
    """A circuit with a mid-circuit measurement, followed by more ops."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)
    qc.measure(1, 1)
    return qc


def _if_else_circuit_loose_clbit() -> tuple[QuantumCircuit, Clbit]:
    """Circuit with a loose Clbit used as an IfElseOp condition."""
    qc = QuantumCircuit(1)
    qc.add_bits([Clbit()])
    condition_bit = qc.clbits[0]
    qc.measure(0, condition_bit)
    with qc.if_test((condition_bit, 1)):
        qc.x(0)
    return qc, condition_bit


def _if_else_circuit_existing_creg() -> tuple[QuantumCircuit, Clbit]:
    """Circuit with an existing ClassicalRegister used as an IfElseOp condition."""
    cr = ClassicalRegister(1, "a")
    qc = QuantumCircuit(QuantumRegister(1, "q"), cr)
    condition_bit = cr[0]
    qc.measure(0, condition_bit)
    with qc.if_test((condition_bit, 1)):
        qc.x(0)
    return qc, condition_bit


def _if_else_circuit_for_verbatim() -> QuantumCircuit:
    """Circuit with an IfElseOp — used to check verbatim wrapping preserves it."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
    return qc


def _build_cx_and_p() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.p(0.5, 0)
    return qc


def _build_rxx_and_r() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rxx(0.3, 0, 1)
    qc.r(0.7, 0.1, 0)
    return qc


def _build_cx_only_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    return qc


def _build_native_gates_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.rx(0.5, 1)
    return qc


def _build_already_shimmed_circuit() -> QuantumCircuit:
    return PassManager([RenameGates()]).run(_build_cx_only_circuit())


def _build_if_else_body_with_cx_and_p() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.cx(0, 1)
        qc.p(0.5, 1)
    return qc


def _build_box_body_with_cx_and_p() -> QuantumCircuit:
    inner = QuantumCircuit(2)
    inner.cx(0, 1)
    inner.p(0.5, 0)
    outer = QuantumCircuit(2)
    outer.append(BraketVerbatimBox(inner), [0, 1])
    return outer


def _build_bell_with_rxx() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rxx(0.5, 0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _build_bell() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _build_plain_box() -> QuantumCircuit:
    inner = QuantumCircuit(2)
    inner.h(0)
    inner.cx(0, 1)
    qc = QuantumCircuit(2)
    qc.append(BoxOp(inner, label="plain"), [0, 1])
    return qc


def _get_if_else(circuit: QuantumCircuit) -> IfElseOp:
    return next(instr.operation for instr in circuit.data if isinstance(instr.operation, IfElseOp))


def _if_else_body_ops(circuit: QuantumCircuit) -> list:
    """Extract the true-branch ops from an IfElseOp inside ``circuit``."""
    op = next(instr.operation for instr in circuit.data if isinstance(instr.operation, IfElseOp))
    return list(op.blocks[0].data)


def _box_body_ops(circuit: QuantumCircuit) -> list:
    """Extract the body ops from a BoxOp inside ``circuit``."""
    op = next(instr.operation for instr in circuit.data if isinstance(instr.operation, BoxOp))
    return list(op.blocks[0].data)


def _assert_contents(source: str, expected_present: list[str], expected_absent: list[str]) -> None:
    for s in expected_present:
        assert s in source
    for s in expected_absent:
        assert s not in source


def _dumps_with_passes(circuit: QuantumCircuit, *, basis_gates: list[str], verbatim: bool) -> str:
    """Run the passes and dump to OQ3"""
    pm = PassManager()
    pm.append(ConsolidateClbits())
    pm.append(MoveMeasurementsToEnd())
    if verbatim:
        pm.append(WrapInVerbatimBox())
    pm.append(RenameGates())
    compiled = pm.run(circuit)
    return qasm3.dumps(
        compiled,
        includes=[],
        basis_gates=basis_gates,
        disable_constants=True,
        annotation_handlers=_BRAKET_ANNOTATION_HANDLERS,
    )


@pytest.mark.parametrize(
    "build_circuit,expected_creg_names,expected_num_clbits",
    [
        (_two_cregs_circuit, ["b"], 2),
        (_default_creg_circuit, ["b"], 2),
        (_no_clbits_circuit, [], 0),
        (_output_register_circuit, ["c"], 2),
        (_output_and_scratch_circuit, ["first", "b"], 4),
        (_shadow_b_circuit, ["b", "b0"], 2),
        (_shadow_b_and_b0_circuit, ["b", "b0", "b1"], 3),
        (_all_output_registers_circuit, ["a", "b"], 2),
    ],
    ids=[
        "two_regs",
        "renames_default_reg",
        "no_clbits_noop",
        "output_register_preserved",
        "output_and_plain_bits",
        "shadow_b_falls_back_to_b0",
        "shadow_b_and_b0_falls_back_to_b1",
        "no_plain_register_when_all_outputs",
    ],
)
def test_consolidate_clbits(
    build_circuit: Callable, expected_creg_names: list[str], expected_num_clbits: int
) -> None:
    result = PassManager([ConsolidateClbits()]).run(build_circuit())
    assert [creg.name for creg in result.cregs] == expected_creg_names
    assert result.num_clbits == expected_num_clbits


@pytest.mark.parametrize(
    "build_circuit",
    [_if_else_circuit_loose_clbit, _if_else_circuit_existing_creg],
    ids=["loose_clbit", "existing_creg"],
)
def test_consolidate_clbits_preserves_if_else_condition(build_circuit: Callable) -> None:
    """The IfElseOp condition must still point at the same Clbit after consolidation."""
    circuit, condition_bit = build_circuit()
    result = PassManager([ConsolidateClbits()]).run(circuit)
    cond_clbit, cond_value = _get_if_else(result).condition
    assert cond_clbit == condition_bit
    assert cond_value == 1
    # The Clbit is now a member of the "b" register.
    assert any(condition_bit in creg for creg in result.cregs if creg.name == "b")


@pytest.mark.parametrize(
    "build_circuit,dynamic_circuits_supported,expected_op_order",
    [
        (_mid_measure_circuit, False, ["h", "cx", "measure", "measure"]),
        (_mid_measure_circuit, True, ["h", "measure", "cx", "measure"]),
        (_bell_circuit, False, ["h", "cx", "measure", "measure"]),
    ],
    ids=["reorders_mid_measure", "dynamic_circuits_noop", "already_at_end_noop"],
)
def test_move_measurements_to_end(
    build_circuit: Callable, dynamic_circuits_supported: bool, expected_op_order: list[str]
) -> None:
    result = PassManager([MoveMeasurementsToEnd(dynamic_circuits_supported)]).run(build_circuit())
    assert [instr.operation.name for instr in result.data] == expected_op_order


@pytest.mark.parametrize(
    "build_circuit,dynamic_circuits_supported,expected_inner_ops,expected_trailing_ops,"
    "expected_metadata",
    [
        (
            _bell_circuit,
            False,
            ["h", "cx"],
            ["measure", "measure"],
            {},
        ),
        (
            _bell_circuit,
            True,
            ["h", "cx", "measure", "measure"],
            [],
            {},
        ),
        (
            _output_register_circuit,
            False,
            ["h", "cx"],
            ["measure", "measure"],
            {"braket_output_variables": {"c": None}},
        ),
    ],
    ids=[
        "measurements_outside_by_default",
        "everything_inside_when_dynamic",
        "preserves_metadata",
    ],
)
def test_wrap_in_verbatim_box(
    build_circuit: Callable,
    dynamic_circuits_supported: bool,
    expected_inner_ops: list[str],
    expected_trailing_ops: list[str],
    expected_metadata: dict,
) -> None:
    result = PassManager([
        WrapInVerbatimBox(dynamic_circuits_supported=dynamic_circuits_supported)
    ]).run(build_circuit())

    top_level_boxes = [instr for instr in result.data if isinstance(instr.operation, BoxOp)]
    assert len(top_level_boxes) == 1
    box_op = top_level_boxes[0].operation
    assert box_op.label == _BRAKET_VERBATIM_BOX_NAME
    assert any(isinstance(a, BraketVerbatim) for a in box_op.annotations)

    op_names = [instr.operation.name for instr in result.data]
    box_idx = op_names.index("box")
    assert op_names[box_idx + 1 :] == expected_trailing_ops

    inner_ops = [instr.operation.name for instr in box_op.blocks[0].data]
    assert inner_ops == expected_inner_ops
    assert result.metadata == expected_metadata


def test_wrap_in_verbatim_box_preserves_if_else_body_when_dynamic() -> None:
    """When dynamic_circuits_supported=True, an IfElseOp is placed inside the verbatim box."""
    result = PassManager([WrapInVerbatimBox(dynamic_circuits_supported=True)]).run(
        _if_else_circuit_for_verbatim()
    )

    top_level_boxes = [instr for instr in result.data if isinstance(instr.operation, BoxOp)]
    assert len(top_level_boxes) == 1
    inner_ops = top_level_boxes[0].operation.blocks[0].data
    assert any(isinstance(instr.operation, IfElseOp) for instr in inner_ops)


@pytest.mark.parametrize(
    "circuit",
    [QuantumCircuit(2, 2), QuantumCircuit()],
    ids=["wires_only", "nothing_at_all"],
)
def test_wrap_in_verbatim_box_empty_circuit(circuit: QuantumCircuit) -> None:
    """An empty circuit yields a single empty verbatim box spanning its wires."""
    result = PassManager([WrapInVerbatimBox()]).run(circuit)

    top_level_boxes = [instr for instr in result.data if isinstance(instr.operation, BoxOp)]
    assert len(top_level_boxes) == 1
    box_instr = top_level_boxes[0]
    assert box_instr.operation.label == _BRAKET_VERBATIM_BOX_NAME
    assert any(isinstance(a, BraketVerbatim) for a in box_instr.operation.annotations)
    assert list(box_instr.qubits) == list(circuit.qubits)
    assert list(box_instr.clbits) == list(circuit.clbits)
    assert list(box_instr.operation.blocks[0].data) == []


@pytest.mark.parametrize(
    "build_circuit,expected_ops,expected_names",
    [
        (
            _build_cx_and_p,
            [_QISKIT_TO_BRAKET_SHIM[CXGate], _QISKIT_TO_BRAKET_SHIM[PhaseGate]],
            ["cnot", "phaseshift"],
        ),
        (
            _build_rxx_and_r,
            [_QISKIT_TO_BRAKET_SHIM[RXXGate], _QISKIT_TO_BRAKET_SHIM[RGate]],
            ["xx", "prx"],
        ),
    ],
    ids=["cx_and_p", "rxx_and_r"],
)
def test_rename_gates_substitutes_shims(
    build_circuit: Callable, expected_ops: list[type], expected_names: list[str]
) -> None:
    """Each Qiskit gate is replaced with its shim, preserving parameters."""
    result = PassManager([RenameGates()]).run(build_circuit())
    ops = [instr.operation for instr in result.data]
    for op, expected_cls in zip(ops, expected_ops, strict=True):
        assert isinstance(op, expected_cls)
    assert [op.name for op in ops] == expected_names


@pytest.mark.parametrize(
    "build_circuit,expected_names",
    [
        (_build_native_gates_circuit, ["h", "rx"]),
        (_build_already_shimmed_circuit, ["cnot"]),
    ],
    ids=["already_native", "already_shimmed"],
)
def test_rename_gates_is_noop_when_nothing_to_rename(
    build_circuit: Callable, expected_names: list[str]
) -> None:
    """Gates already using Braket-native names (or already shimmed) pass through unchanged."""
    result = PassManager([RenameGates()]).run(build_circuit())
    assert [instr.operation.name for instr in result.data] == expected_names


@pytest.mark.parametrize(
    "build_circuit,extract_body_ops",
    [
        (_build_if_else_body_with_cx_and_p, _if_else_body_ops),
        (_build_box_body_with_cx_and_p, _box_body_ops),
    ],
    ids=["if_else_body", "box_body"],
)
def test_rename_gates_recurses_into_block_bodies(
    build_circuit: Callable, extract_body_ops: Callable
) -> None:
    """Gates nested inside IfElseOp or BoxOp block bodies are renamed too."""
    result = PassManager([RenameGates()]).run(build_circuit())
    body_ops = extract_body_ops(result)
    assert isinstance(body_ops[0].operation, _QISKIT_TO_BRAKET_SHIM[CXGate])
    assert isinstance(body_ops[1].operation, _QISKIT_TO_BRAKET_SHIM[PhaseGate])


def test_rename_gates_recurses_into_deeply_nested_blocks() -> None:
    """Recursion handles BoxOp nested inside IfElseOp's true-branch body."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        inner = QuantumCircuit(2)
        inner.cx(0, 1)
        inner.p(0.5, 0)
        qc.append(BoxOp(inner, label="nested"), [0, 1])

    result = PassManager([RenameGates()]).run(qc)
    if_else = next(i.operation for i in result.data if isinstance(i.operation, IfElseOp))
    inner_box = next(i.operation for i in if_else.blocks[0].data if isinstance(i.operation, BoxOp))
    ops = list(inner_box.blocks[0].data)
    assert isinstance(ops[0].operation, _QISKIT_TO_BRAKET_SHIM[CXGate])
    assert isinstance(ops[1].operation, _QISKIT_TO_BRAKET_SHIM[PhaseGate])


def test_rename_gates_preserves_label() -> None:
    """A user-set ``label`` on the original gate survives the shim substitution."""
    qc = QuantumCircuit(2)
    labeled = CXGate(label="my_cx")
    qc.append(labeled, [0, 1])
    result = PassManager([RenameGates()]).run(qc)
    shimmed = result.data[0].operation
    assert isinstance(shimmed, _QISKIT_TO_BRAKET_SHIM[CXGate])
    assert isinstance(shimmed, CXGate)  # narrow for type-checker attribute access below
    assert shimmed.label == "my_cx"


def test_shim_class_for_returns_none_for_already_shimmed_op() -> None:
    """The idempotence guard: a shim instance is recognised via its MRO and skipped.

    Called directly because ``PassManager``'s ``dag_to_circuit`` round-trip
    strips shim-gate identity, so the pass never sees a shim in practice.
    """
    assert _shim_class_for(_QISKIT_TO_BRAKET_SHIM[CXGate]()) is None


def test_rename_gates_preserves_parameters() -> None:
    """Shim retains the original gate's numeric parameters."""
    qc = QuantumCircuit(2)
    qc.rxx(1.5, 0, 1)
    qc.p(0.75, 0)
    result = PassManager([RenameGates()]).run(qc)
    ops = [instr.operation for instr in result.data]
    assert ops[0].params == [1.5]
    assert ops[1].params == [0.75]


def test_rename_gates_covers_every_shim() -> None:
    """Every Qiskit gate registered in the shim map is renamed to its Braket name."""
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.p(0.1, 0)
    qc.tdg(0)
    qc.sdg(0)
    qc.sx(0)
    qc.sxdg(0)
    qc.rxx(0.2, 0, 1)
    qc.ryy(0.3, 0, 1)
    qc.rzz(0.4, 0, 1)
    qc.id(0)
    qc.ccx(0, 1, 2)
    qc.cp(0.5, 0, 1)
    qc.r(0.6, 0.7, 0)
    qc.append(GlobalPhaseGate(0.8), [])

    result = PassManager([RenameGates()]).run(qc)
    for instr in result.data:
        expected_shim = next(
            shim
            for qiskit_cls, shim in _QISKIT_TO_BRAKET_SHIM.items()
            if isinstance(instr.operation, qiskit_cls)
        )
        assert isinstance(instr.operation, expected_shim)


def test_braket_verbatim_serializer_dumps_pragma_payload() -> None:
    """dump() emits the pragma-verbatim payload for a BraketVerbatim annotation."""
    assert BraketVerbatimSerializer().dump(BraketVerbatim()) == "pragma braket verbatim"


def test_braket_verbatim_annotations_are_value_equal() -> None:
    """Any two ``BraketVerbatim`` instances compare equal and share a hash.

    Required for ``BoxOp.__eq__`` on Qiskit versions that compare annotations
    by value: hand-built and library-built verbatim boxes must round-trip
    through equality checks (e.g. in tests that assert ``to_qiskit(...) == qc``).
    """
    a, b = BraketVerbatim(), BraketVerbatim()
    assert a == b
    assert hash(a) == hash(b)
    assert a != BraketVerbatim.namespace


@pytest.mark.parametrize(
    "label_kwarg,expected_label",
    [({}, _BRAKET_VERBATIM_BOX_NAME), ({"label": "custom_verbatim"}, "custom_verbatim")],
    ids=["default_label", "custom_label"],
)
def test_braket_verbatim_box_label_and_annotation(label_kwarg: dict, expected_label: str) -> None:
    """BraketVerbatimBox always attaches the annotation; label defaults or accepts an override."""
    box = BraketVerbatimBox(QuantumCircuit(2), **label_kwarg)
    assert box.label == expected_label
    assert any(isinstance(a, BraketVerbatim) for a in box.annotations)


@pytest.mark.parametrize(
    "source,qubit_labels,expected_present,expected_absent,expect_unchanged",
    [
        (
            "OPENQASM 3.0;\nbit[2] c;\nqubit[2] q;\nh q[0];\ncnot q[0], q[1];\n",
            [3, 7],
            ["$3", "$7"],
            ["qubit["],
            False,
        ),
        (
            "OPENQASM 3.0;\nbit[2] c;\nh $0;\ncnot $0, $1;\n",
            [3, 7],
            ["$3", "$7"],
            ["$0;", "$1;"],
            False,
        ),
        (
            "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\n",
            None,
            [],
            [],
            True,
        ),
    ],
    ids=["virtual_form", "layout_aware_form", "noop_when_none"],
)
def test_remap_qubits(
    source: str,
    qubit_labels: list[int] | None,
    expected_present: list[str],
    expected_absent: list[str],
    expect_unchanged: bool,
) -> None:
    result = _remap_qubits(source, qubit_labels)
    if expect_unchanged:
        assert result == source
    _assert_contents(result, expected_present, expected_absent)


def test_remap_qubits_raises_on_label_length_mismatch() -> None:
    """Length mismatch against a declared virtual register raises ``ValueError``."""
    source = "OPENQASM 3.0;\nqubit[3] q;\nh q[0];\n"
    with pytest.raises(
        ValueError, match=r"qubit_labels length \(2\) does not match .*qubit count \(3\)"
    ):
        _remap_qubits(source, [3, 7])


def test_remap_qubits_rejects_multiple_quantum_registers() -> None:
    """Braket doesn't support multi-register circuits; ``_remap_qubits`` raises."""
    source = "OPENQASM 3.0;\nqubit[2] a;\nqubit[2] bq;\nh a[0];\ncx a[0], bq[1];\n"
    with pytest.raises(
        ValueError, match=r"Braket does not support multiple quantum registers.*'a'.*'bq'"
    ):
        _remap_qubits(source, [10, 11, 12, 13])


def test_remap_qubits_rejects_multiple_registers_without_labels() -> None:
    """Multi-register rejection fires even when ``qubit_labels`` is not provided."""
    source = "OPENQASM 3.0;\nqubit[2] a;\nqubit[2] bq;\nh a[0];\n"
    with pytest.raises(ValueError, match=r"Braket does not support multiple quantum registers"):
        _remap_qubits(source, None)


def test_remap_qubits_layout_aware_out_of_range_raises_value_error() -> None:
    """Out-of-range ``$N`` in layout-aware source produces a ``ValueError``, not IndexError."""
    source = "OPENQASM 3.0;\nh $0;\ncnot $0, $5;\n"
    with pytest.raises(
        ValueError, match=r"qubit_labels length \(2\) is too short for physical qubit reference \$5"
    ):
        _remap_qubits(source, [10, 11])


@pytest.mark.parametrize(
    "source,output_names,expected_present,expected_absent",
    [
        (
            "OPENQASM 3.0;\nbit[2] c;\n@braket_verbatim pragma braket verbatim\nbox {\n  h $0;\n}\n",
            (),
            ["#pragma braket verbatim", "box {"],
            ["@braket_verbatim"],
        ),
        (
            "OPENQASM 3.0;\nbit[2] c;\nbox {\n  h $0;\n}\n",
            (),
            ["box {"],
            ["#pragma braket verbatim", "@braket_verbatim"],
        ),
        (
            "OPENQASM 3.0;\n  box {\n    h $0;\n  }\n",
            (),
            ["box {"],
            ["  "],
        ),
        (
            "OPENQASM 3.0;\nh q[0];\n",
            (),
            ["OPENQASM 3.0;", "h q[0];"],
            ["#pragma"],
        ),
        (
            "OPENQASM 3.0;\ninput float[64] theta;\n",
            (),
            ["float theta;"],
            ["float[64]"],
        ),
        (
            "OPENQASM 3.0;\nbit[2] c;\nbit[1] scratch;\n",
            ("c",),
            ["output bit[2] c;", "bit[1] scratch;"],
            ["output bit[1] scratch;"],
        ),
    ],
    ids=[
        "annotated_box_gets_pragma",
        "plain_box_passes_through",
        "strips_indentation",
        "noop_without_box",
        "replaces_float64",
        "restores_output_declaration",
    ],
)
def test_normalize_formatting(
    source: str,
    output_names: tuple[str, ...],
    expected_present: list[str],
    expected_absent: list[str],
) -> None:
    _assert_contents(_normalize_formatting(source, output_names), expected_present, expected_absent)


def test_post_process_oq3() -> None:
    """The two helpers compose to yield remap + normalize on a single source."""
    source = "OPENQASM 3.0;\ninput float[64] theta;\nbit[2] c;\nqubit[2] q;\ncnot q[0], q[1];\n"
    result = _normalize_formatting(_remap_qubits(source, [0, 1]), output_names=("c",))
    _assert_contents(
        result,
        ["cnot $0, $1;", "output bit[2] c;", "float theta;"],
        ["float[64]", "qubit["],
    )


@pytest.mark.parametrize(
    "build_circuit,basis_gates,verbatim,qubit_labels,expected_present,expected_absent",
    [
        (
            _build_bell_with_rxx,
            ["h", "cnot", "xx"],
            False,
            [3, 7],
            ["h $3;", "cnot $3, $7;", "xx(0.5) $3, $7;", "bit[2] b;"],
            ["cx ", "rxx", "qubit["],
        ),
        (
            _build_bell,
            ["h", "cnot"],
            True,
            [0, 1],
            ["#pragma braket verbatim", "box {", "h $0;", "cnot $0, $1;"],
            ["@braket_verbatim", "cx "],
        ),
        (
            _build_plain_box,
            ["h", "cnot"],
            False,
            None,
            ["box {", "h q[0];", "cnot q[0], q[1];"],
            ["#pragma braket verbatim", "@braket_verbatim"],
        ),
    ],
    ids=["renames_and_remaps", "verbatim_pragma", "plain_box_preserved"],
)
def test_post_process_oq3_end_to_end(
    build_circuit: Callable,
    basis_gates: list[str],
    verbatim: bool,
    qubit_labels: list[int] | None,
    expected_present: list[str],
    expected_absent: list[str],
) -> None:
    """End-to-end: qasm3.dumps → remap + normalize produces Braket-compatible output."""
    source = _dumps_with_passes(build_circuit(), basis_gates=basis_gates, verbatim=verbatim)
    result = _normalize_formatting(_remap_qubits(source, qubit_labels))
    _assert_contents(result, expected_present, expected_absent)
