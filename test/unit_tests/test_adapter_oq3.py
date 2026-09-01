"""Tests for the adapter OQ3 output path: ``to_oq3`` and ``compile_to_oq3``."""

from collections.abc import Callable

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import (
    BoxOp,
    ClassicalRegister,
    Clbit,
    Measure,
    Parameter,
    ParameterVector,
    QuantumRegister,
)
from qiskit.circuit.library import CXGate, HGate
from qiskit.transpiler import Target

from braket.devices import LocalSimulator
from braket.ir.openqasm import Program
from qiskit_braket_provider.providers.adapter import compile_to_oq3, to_oq3
from qiskit_braket_provider.providers.gate_mappings import _BRAKET_VERBATIM_BOX_NAME


@pytest.fixture
def bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


@pytest.fixture
def ghz_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))
    return qc


@pytest.fixture
def sim() -> LocalSimulator:
    return LocalSimulator("braket_sv")


def _assert_contents(source: str, expected_present: list[str], expected_absent: list[str]) -> None:
    for s in expected_present:
        assert s in source
    for s in expected_absent:
        assert s not in source


def _output_circuit(output_names: tuple[str, ...], sizes: tuple[int, ...]) -> QuantumCircuit:
    """Build a circuit with each name mapped to a ClassicalRegister of the given size."""
    total = sum(sizes)
    qc = QuantumCircuit(total)
    q_idx = 0
    for name, size in zip(output_names, sizes, strict=True):
        creg = ClassicalRegister(size, name)
        qc.add_register(creg)
        for i in range(size):
            qc.measure(q_idx, creg[i])
            q_idx += 1
    qc.metadata = {"braket_output_variables": dict.fromkeys(output_names)}
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


def _bell_circuit_target() -> Target:
    target = Target(num_qubits=2)
    target.add_instruction(HGate(), name="h")
    target.add_instruction(CXGate(), name="cx")
    target.add_instruction(Measure(), name="measure")
    return target


def _bell_with_verbatim_boxop() -> QuantumCircuit:
    inner = QuantumCircuit(2)
    inner.h(0)
    inner.cx(0, 1)
    outer = QuantumCircuit(2, 2)
    outer.append(BoxOp(inner, label=_BRAKET_VERBATIM_BOX_NAME), [0, 1])
    outer.measure([0, 1], [0, 1])
    return outer


@pytest.mark.parametrize(
    "qubit_labels,expected_present,expected_absent",
    [
        (None, ["qubit[2] q;", "q[0]", "q[1]"], []),
        ([0, 4], ["$0", "$4"], ["qubit[", "q["]),
    ],
    ids=["no_labels", "non_contiguous_labels"],
)
def test_to_oq3_qubit_remapping(
    bell_circuit: QuantumCircuit,
    qubit_labels: list[int] | None,
    expected_present: list[str],
    expected_absent: list[str],
) -> None:
    oq3 = to_oq3(bell_circuit, basis_gates=["h", "cx"], qubit_labels=qubit_labels)
    _assert_contents(oq3, expected_present, expected_absent)


def test_to_oq3_verbatim_wrapping(bell_circuit: QuantumCircuit) -> None:
    oq3 = to_oq3(
        bell_circuit,
        basis_gates=["h", "cx"],
        qubit_labels=[0, 1],
        should_wrap_verbatim=True,
    )
    _assert_contents(
        oq3,
        ["#pragma braket verbatim", "box {", "h $0;", "cnot $0, $1;"],
        [],
    )


def test_to_oq3_consolidates_classical_registers() -> None:
    qr = QuantumRegister(3, "q")
    cr1 = ClassicalRegister(2, "meas1")
    cr2 = ClassicalRegister(1, "meas2")
    qc = QuantumCircuit(qr, cr1, cr2)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(0, cr1[0])
    qc.measure(1, cr1[1])
    qc.measure(2, cr2[0])

    oq3 = to_oq3(qc, basis_gates=["h", "cx"], qubit_labels=[0, 1, 7])
    _assert_contents(
        oq3,
        [
            "bit[3] b;",
            "b[0] = measure $0;",
            "b[1] = measure $1;",
            "b[2] = measure $7;",
        ],
        ["meas1", "meas2"],
    )


def test_to_oq3_single_creg_unchanged(bell_circuit: QuantumCircuit) -> None:
    assert "bit[2] b;" in to_oq3(bell_circuit, basis_gates=["h", "cx"])


@pytest.mark.parametrize(
    "build_params,build_circuit,expected_present",
    [
        (
            lambda: (Parameter("theta"),),
            lambda qc, params: (qc.rx(params[0], 0), qc.measure(0, 0)),
            ["input float theta;", "rx(theta) $5;"],
        ),
        (
            lambda: ParameterVector("p", 2),
            lambda qc, params: (
                qc.rx(params[0], 0),
                qc.ry(params[1], 1),
                qc.measure([0, 1], [0, 1]),
            ),
            ["input float", "rx(", "ry("],
        ),
    ],
    ids=["parameter", "parameter_vector"],
)
def test_to_oq3_parameterized_circuits(
    build_params: Callable, build_circuit: Callable, expected_present: list[str]
) -> None:
    params = build_params()
    num_qubits = 1 if len(params) == 1 else 2
    qc = QuantumCircuit(num_qubits, num_qubits)
    build_circuit(qc, params)
    qubit_labels = [5] if num_qubits == 1 else None
    oq3 = to_oq3(
        qc,
        basis_gates=[
            str(instr.operation.name) for instr in qc.data if instr.operation.name != "measure"
        ],
        qubit_labels=qubit_labels,
    )
    _assert_contents(oq3, expected_present, [])


def test_to_oq3_auto_detects_basis_gates(bell_circuit: QuantumCircuit) -> None:
    oq3 = to_oq3(bell_circuit)
    _assert_contents(oq3, ["h ", "cnot "], ["gate "])


def test_to_oq3_skips_renaming_for_native_gates() -> None:
    """When all gates are already Braket-native, no renaming occurs."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.rx(0.5, 1)
    qc.measure([0, 1], [0, 1])
    oq3 = to_oq3(qc, basis_gates=["h", "rx"])
    _assert_contents(oq3, ["h ", "rx(0.5)"], ["cnot", "phaseshift"])


@pytest.mark.parametrize(
    "build_circuit,expected_present,expected_absent",
    [
        (
            lambda qc: (qc.sx(0), qc.sdg(0), qc.tdg(1), qc.cx(0, 1)),
            ["v ", "si ", "ti ", "cnot "],
            ["sx ", "sdg ", "tdg ", "cx "],
        ),
        (
            lambda qc: (qc.rxx(0.5, 0, 1), qc.ryy(0.3, 0, 1), qc.rzz(0.7, 0, 1)),
            ["xx(0.5)", "yy(0.3)", "zz(0.7)"],
            ["rxx", "ryy", "rzz"],
        ),
        (lambda qc: (qc.id(0),), ["i "], []),
        (lambda qc: (qc.p(0.5, 0),), ["phaseshift(0.5)"], []),
        (lambda qc: (qc.cp(0.5, 0, 1),), ["cphaseshift(0.5)"], []),
    ],
    ids=["sx_sdg_tdg_cx", "rxx_ryy_rzz", "id", "p", "cp"],
)
def test_gate_renaming_in_output(
    build_circuit: Callable, expected_present: list[str], expected_absent: list[str]
) -> None:
    qc = QuantumCircuit(2, 2)
    build_circuit(qc)
    qc.measure([0, 1], [0, 1])
    _assert_contents(compile_to_oq3(qc), expected_present, expected_absent)


def test_compile_to_oq3_list_input(bell_circuit: QuantumCircuit) -> None:
    results = compile_to_oq3([bell_circuit, bell_circuit])
    assert isinstance(results, list)
    assert len(results) == 2
    assert all("OPENQASM 3.0;" in r for r in results)


@pytest.mark.parametrize(
    "build_circuit,compile_kwargs,expected_present,expected_absent",
    [
        (
            lambda _bell: _bell,
            {"verbatim": True, "qubit_labels": [0, 1]},
            ["#pragma braket verbatim", "box {"],
            [],
        ),
        (
            lambda _bell: _bell,
            {"target": _bell_circuit_target(), "qubit_labels": [0, 1]},
            ["OPENQASM 3.0;", "h ", "cnot ", "#pragma braket verbatim", "box {"],
            [],
        ),
        (
            lambda _bell: _bell_with_verbatim_boxop(),
            {"qubit_labels": [0, 1]},
            ["h ", "cnot "],
            [],
        ),
    ],
    ids=["verbatim_flag", "target", "existing_verbatim_box"],
)
def test_compile_to_oq3_verbatim_wrapping_triggers(
    bell_circuit: QuantumCircuit,
    build_circuit: Callable,
    compile_kwargs: dict,
    expected_present: list[str],
    expected_absent: list[str],
) -> None:
    _assert_contents(
        compile_to_oq3(build_circuit(bell_circuit), **compile_kwargs),
        expected_present,
        expected_absent,
    )


def test_compile_to_oq3_non_contiguous_qubit_labels(ghz_circuit: QuantumCircuit) -> None:
    _assert_contents(
        compile_to_oq3(ghz_circuit, verbatim=True, qubit_labels=[0, 4, 7]),
        ["$0", "$4", "$7"],
        ["q["],
    )


@pytest.mark.parametrize(
    "verbatim,qubit_labels,build_circuit",
    [
        (False, None, lambda qc: (qc.h(0), qc.cx(0, 1))),
        (True, [0, 1], lambda qc: (qc.h(0), qc.cx(0, 1))),
        (False, [0, 1], lambda qc: (qc.sx(0), qc.sdg(1), qc.cx(0, 1))),
    ],
    ids=["default", "verbatim", "renamed_gates"],
)
def test_compile_to_oq3_accepted_by_braket_simulator(
    sim: LocalSimulator,
    verbatim: bool,
    qubit_labels: list[int] | None,
    build_circuit: Callable,
) -> None:
    qc = QuantumCircuit(2, 2)
    build_circuit(qc)
    qc.measure([0, 1], [0, 1])

    oq3 = compile_to_oq3(qc, verbatim=verbatim, qubit_labels=qubit_labels)
    result = sim.run(Program(source=oq3), shots=100)
    assert result.result().measurements.shape == (100, 2)


def test_compile_to_oq3_raises_on_invalid_input() -> None:
    with pytest.raises(TypeError):
        compile_to_oq3("not a circuit")


def test_compile_to_oq3_raises_on_conflicting_options(bell_circuit: QuantumCircuit) -> None:
    target = Target(num_qubits=2)
    target.add_instruction(HGate(), name="h")
    target.add_instruction(CXGate(), name="cx")
    with pytest.raises(ValueError):
        compile_to_oq3(bell_circuit, target=target, basis_gates=["h", "cx"])


@pytest.mark.parametrize(
    "build_circuit,compile_kwargs,expected_present,expected_absent",
    [
        (
            lambda: _output_circuit(("c",), (2,)),
            {},
            ["output bit[2] c;"],
            ["bit[2] b;"],
        ),
        (
            lambda: _output_circuit(("first", "second"), (2, 1)),
            {},
            ["output bit[2] first;", "output bit[1] second;"],
            [],
        ),
        (
            _output_and_scratch_circuit,
            {},
            ["output bit[2] first;", "bit[2] b;"],
            ["scratch"],
        ),
        (
            lambda: _output_circuit(("c",), (2,)),
            {"verbatim": True, "qubit_labels": [0, 1]},
            ["output bit[2] c;", "#pragma braket verbatim", "box {"],
            [],
        ),
        (
            _shadow_b_circuit,
            {},
            ["output bit[1] b;", "bit[1] b0;"],
            [],
        ),
        (
            _shadow_b_and_b0_circuit,
            {},
            ["output bit[1] b;", "output bit[1] b0;", "bit[1] b1;"],
            [],
        ),
    ],
    ids=[
        "single_output",
        "multiple_outputs",
        "output_with_plain_bits",
        "output_survives_verbatim",
        "plain_name_avoids_shadow",
        "plain_name_avoids_double_shadow",
    ],
)
def test_compile_to_oq3_output_declarations(
    build_circuit: Callable,
    compile_kwargs: dict,
    expected_present: list[str],
    expected_absent: list[str],
) -> None:
    _assert_contents(
        compile_to_oq3(build_circuit(), **compile_kwargs), expected_present, expected_absent
    )


def test_compile_to_oq3_output_declaration_precedes_verbatim_box() -> None:
    """Output declarations stay outside the verbatim box."""
    qc = _output_circuit(("c",), (2,))
    oq3 = compile_to_oq3(qc, verbatim=True, qubit_labels=[0, 1])
    lines = oq3.split("\n")
    output_idx = next(i for i, ln in enumerate(lines) if ln.startswith("output bit["))
    box_idx = next(i for i, ln in enumerate(lines) if ln == "box {")
    assert output_idx < box_idx


@pytest.mark.parametrize(
    "metadata",
    [{}, {"braket_output_variables": {}}, {"unrelated": "value"}],
    ids=["empty", "empty_output_map", "unrelated_key"],
)
def test_compile_to_oq3_no_output_metadata(bell_circuit: QuantumCircuit, metadata: dict) -> None:
    """Without output metadata, no bit declaration is prefixed with `output`."""
    bell_circuit.metadata = metadata
    assert "output bit[" not in compile_to_oq3(bell_circuit)
