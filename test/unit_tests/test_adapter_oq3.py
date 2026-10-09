"""Tests for the adapter OQ3 output path: ``to_oq3`` and ``compile_to_oq3``."""

from collections.abc import Callable
from unittest.mock import Mock

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import (
    BoxOp,
    ClassicalRegister,
    IfElseOp,
    Measure,
)
from qiskit.circuit.library import CXGate, HGate, XGate
from qiskit.transpiler import Target

from braket.circuits.serialization import (
    IRType,
    OpenQASMSerializationProperties,
    QubitReferenceType,
)
from braket.devices import LocalSimulator
from braket.ir.openqasm import Program
from qiskit_braket_provider.providers.adapter import (
    _has_control_flow,
    compile_to_oq3,
    to_braket,
    to_oq3,
)
from qiskit_braket_provider.providers.gate_mappings import _BRAKET_VERBATIM_BOX_NAME
from qiskit_braket_provider.providers.target import aws_device_to_target

from .mocks import mock_iqm_device


def _bell_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _ghz_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure(range(3), range(3))
    return qc


def _sx_sdg_cx_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.sx(0)
    qc.sdg(1)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _rccx_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.rccx(0, 1, 2)
    return qc


def _rccx_in_if_else_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3, 1)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.rccx(0, 1, 2)
    return qc


def _prx_circuit() -> QuantumCircuit:
    """Two-qubit circuit already in IQM's native gates (``RGate`` is ``prx``)."""
    qc = QuantumCircuit(2, 2)
    qc.r(0.5, 0, 0)
    qc.r(0.25, 0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def _if_else_circuit() -> QuantumCircuit:
    """Small circuit with a top-level ``IfElseOp``."""
    true_body = QuantumCircuit(1, 1)
    true_body.x(0)

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.append(IfElseOp((qc.clbits[0], 1), true_body, None), [0], [0])
    return qc


def _if_else_target() -> Target:
    """Target that supports h, x, measure, and if_else on a single qubit."""
    target = Target(num_qubits=1)
    target.add_instruction(HGate(), name="h")
    target.add_instruction(XGate(), name="x")
    target.add_instruction(Measure(), name="measure")
    target.add_instruction(IfElseOp, name="if_else")
    return target


def _mock_non_iqm_device() -> Mock:
    """The IQM mock with a non-IQM (Rigetti) device ARN."""
    device = mock_iqm_device()
    device.arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3"
    return device


def _iqm_device_kwargs() -> dict:
    return {"braket_device": mock_iqm_device()}


def _non_iqm_device_kwargs() -> dict:
    return {"braket_device": _mock_non_iqm_device()}


def _iqm_target_kwargs() -> dict:
    device = mock_iqm_device()
    return {
        "target": aws_device_to_target(device),
        "qubit_labels": sorted(device.topology_graph.nodes),
    }


def _non_iqm_target_kwargs() -> dict:
    device = _mock_non_iqm_device()
    return {
        "target": aws_device_to_target(device),
        "qubit_labels": sorted(device.topology_graph.nodes),
    }


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


def test_to_oq3_auto_basis_gates() -> None:
    """Omitting ``basis_gates`` falls back to the default Braket gate set."""
    oq3 = to_oq3(_bell_circuit())
    _assert_contents(oq3, ["h ", "cnot "], ["gate "])


@pytest.mark.parametrize(
    ("circuit_factory", "basis_gates"),
    [
        (_rccx_circuit, None),
        (_rccx_in_if_else_circuit, None),
        (_bell_circuit, ["h"]),
    ],
    ids=["default_basis", "nested_in_if_else", "explicit_basis"],
)
def test_to_oq3_raises_on_gates_outside_basis(
    circuit_factory: Callable[[], QuantumCircuit], basis_gates: list[str] | None
) -> None:
    with pytest.raises(ValueError, match="not in the basis gate set"):
        to_oq3(circuit_factory(), basis_gates=basis_gates)


def test_to_oq3_accepts_control_flow_with_basis_gates() -> None:
    oq3 = to_oq3(_if_else_circuit(), basis_gates=["h", "x"])

    _assert_contents(oq3, ["if (b[0]) {", "x "], ["gate "])


@pytest.mark.parametrize(
    "circuit_factory,expected",
    [(_if_else_circuit, True), (_bell_circuit, False)],
    ids=["if_else_circuit", "plain_circuit"],
)
def test_has_control_flow(circuit_factory: Callable[[], QuantumCircuit], expected: bool) -> None:
    assert _has_control_flow(circuit_factory()) is expected


@pytest.mark.parametrize(
    ("device_factory", "should_raise"),
    [(mock_iqm_device, True), (_mock_non_iqm_device, False)],
    ids=["iqm", "non_iqm"],
)
def test_compile_to_oq3_control_flow_on_native_path_raises_only_for_iqm(
    device_factory: Callable[[], Mock], should_raise: bool
) -> None:
    target = aws_device_to_target(device_factory())

    if should_raise:
        with pytest.raises(ValueError, match="'if' statements on IQM devices"):
            compile_to_oq3(_if_else_circuit(), target=target)
    else:
        oq3 = compile_to_oq3(_if_else_circuit(), target=target)
        assert "if (b[0]) {" in oq3


def test_compile_to_oq3_wraps_verbatim_when_explicit_even_with_control_flow() -> None:
    """``verbatim=True`` is honored for control-flow circuits, even on the native path."""
    oq3 = compile_to_oq3(_if_else_circuit(), target=_if_else_target(), verbatim=True)

    assert "#pragma braket verbatim" in oq3
    assert "box {" in oq3


def test_compile_to_oq3_verbatim_keeps_mid_circuit_measurement_in_box() -> None:
    qc = QuantumCircuit(1, 2)
    qc.measure(0, 0)
    qc.x(0)
    qc.measure(0, 1)

    oq3 = compile_to_oq3(qc, verbatim=True)

    assert "box {\nb[0] = measure q[0];\nx q[0];\n}\nb[1] = measure q[0];" in oq3


def test_compile_to_oq3_verbatim_with_device_wider_than_circuit() -> None:
    """Verbatim skips widening, so only the first circuit-width device labels are used."""
    oq3 = compile_to_oq3(_prx_circuit(), braket_device=mock_iqm_device(), verbatim=True)

    _assert_contents(oq3, ["prx(0.5, 0.0) $1;", "prx(0.25, 0.0) $2;"], ["$3", "$4", "$5"])


@pytest.mark.parametrize(
    ("device_factory", "expected_match"),
    [(mock_iqm_device, False), (_mock_non_iqm_device, True)],
    ids=["iqm", "non_iqm"],
)
def test_compile_to_oq3_matches_to_braket_serialization(
    device_factory: Callable[[], Mock], expected_match: bool
) -> None:
    """``compile_to_oq3`` matches ``to_braket`` serialized to OpenQASM, except on IQM.

    IQM keeps measurements inside the verbatim box, which a Braket ``Circuit`` cannot do.
    """
    device = device_factory()

    braket_circuit = to_braket(_ghz_circuit(), braket_device=device, seed_transpiler=7)
    braket_oq3 = braket_circuit.to_ir(
        IRType.OPENQASM,
        serialization_properties=OpenQASMSerializationProperties(
            qubit_reference_type=QubitReferenceType.PHYSICAL
        ),
    ).source
    oq3 = compile_to_oq3(_ghz_circuit(), braket_device=device, seed_transpiler=7)

    assert (oq3 == braket_oq3.replace("box{", "box {")) is expected_match


@pytest.mark.parametrize(
    ("include_measurement_in_verbatim", "expected_tail"),
    [
        (True, "b[1] = measure q[1];\n}"),
        (False, "}\nb[0] = measure q[0];\nb[1] = measure q[1];"),
    ],
    ids=["included", "excluded"],
)
def test_to_oq3_include_measurement_in_verbatim(
    include_measurement_in_verbatim: bool, expected_tail: str
) -> None:
    oq3 = to_oq3(
        _bell_circuit(),
        should_wrap_verbatim=True,
        include_measurement_in_verbatim=include_measurement_in_verbatim,
    )

    assert oq3.endswith(expected_tail)


@pytest.mark.parametrize(
    ("kwargs_factory", "expected_tail"),
    [
        (_iqm_device_kwargs, "b[1] = measure $2;\n}"),
        (_non_iqm_device_kwargs, "}\nb[0] = measure $1;\nb[1] = measure $2;"),
        (_iqm_target_kwargs, "b[1] = measure $2;\n}"),
        (_non_iqm_target_kwargs, "}\nb[0] = measure $1;\nb[1] = measure $2;"),
    ],
    ids=["iqm_device", "non_iqm_device", "iqm_target", "non_iqm_target"],
)
def test_compile_to_oq3_measurements_in_verbatim_follow_device(
    kwargs_factory: Callable[[], dict], expected_tail: str
) -> None:
    oq3 = compile_to_oq3(_bell_circuit(), **kwargs_factory())

    assert oq3.endswith(expected_tail)


def test_compile_to_oq3_list_input() -> None:
    results = compile_to_oq3([_bell_circuit(), _bell_circuit()])
    assert isinstance(results, list)
    assert len(results) == 2
    assert all("OPENQASM 3.0;" in r for r in results)


def test_compile_to_oq3_accepts_oq3_string_input() -> None:
    """Non-QuantumCircuit inputs (OpenQASM 3 source, ``Program``) are accepted."""
    source = (
        "OPENQASM 3.0;\n"
        "bit[2] b;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "b[0] = measure q[0];\n"
        "b[1] = measure q[1];"
    )
    oq3_from_str = compile_to_oq3(source)
    oq3_from_program = compile_to_oq3(Program(source=source))
    _assert_contents(oq3_from_str, ["OPENQASM 3.0;", "h ", "cnot "], [])
    assert oq3_from_str == oq3_from_program


@pytest.mark.parametrize(
    "circuit_factory,compile_kwargs,expected_present,expected_absent",
    [
        (
            _bell_circuit,
            {"verbatim": True, "qubit_labels": [0, 1]},
            ["#pragma braket verbatim", "box {"],
            [],
        ),
        (
            _bell_circuit,
            {"target": _bell_circuit_target(), "qubit_labels": [0, 1]},
            ["OPENQASM 3.0;", "h ", "cnot ", "#pragma braket verbatim", "box {"],
            [],
        ),
        (
            _bell_with_verbatim_boxop,
            {"qubit_labels": [0, 1]},
            ["h ", "cnot "],
            [],
        ),
        (
            _ghz_circuit,
            {"verbatim": True, "qubit_labels": [0, 4, 7]},
            ["$0", "$4", "$7"],
            ["q["],
        ),
    ],
    ids=["verbatim_flag", "target", "existing_verbatim_box", "non_contiguous_labels"],
)
def test_compile_to_oq3_output(
    circuit_factory: Callable[[], QuantumCircuit],
    compile_kwargs: dict,
    expected_present: list[str],
    expected_absent: list[str],
) -> None:
    _assert_contents(
        compile_to_oq3(circuit_factory(), **compile_kwargs),
        expected_present,
        expected_absent,
    )


@pytest.mark.parametrize(
    "circuit_factory,compile_kwargs,expected_oq3",
    [
        (
            _bell_circuit,
            {},
            (
                "OPENQASM 3.0;\n"
                "bit[2] b;\n"
                "qubit[2] q;\n"
                "h q[0];\n"
                "cnot q[0], q[1];\n"
                "b[0] = measure q[0];\n"
                "b[1] = measure q[1];"
            ),
        ),
        (
            _bell_circuit,
            {"verbatim": True, "qubit_labels": [0, 1]},
            (
                "OPENQASM 3.0;\n"
                "bit[2] b;\n"
                "#pragma braket verbatim\n"
                "box {\n"
                "h $0;\n"
                "cnot $0, $1;\n"
                "}\n"
                "b[0] = measure $0;\n"
                "b[1] = measure $1;"
            ),
        ),
        (
            _sx_sdg_cx_circuit,
            {"qubit_labels": [0, 1]},
            (
                "OPENQASM 3.0;\n"
                "bit[2] b;\n"
                "v $0;\n"
                "si $1;\n"
                "cnot $0, $1;\n"
                "b[0] = measure $0;\n"
                "b[1] = measure $1;"
            ),
        ),
        (
            _bell_circuit,
            {"target": _bell_circuit_target(), "qubit_labels": [0, 1]},
            (
                "OPENQASM 3.0;\n"
                "bit[2] b;\n"
                "#pragma braket verbatim\n"
                "box {\n"
                "h $0;\n"
                "cnot $0, $1;\n"
                "}\n"
                "b[0] = measure $0;\n"
                "b[1] = measure $1;"
            ),
        ),
    ],
    ids=["default", "verbatim", "renamed_gates", "target"],
)
def test_compile_to_oq3_accepted_by_braket_simulator(
    sim: LocalSimulator,
    circuit_factory: Callable[[], QuantumCircuit],
    compile_kwargs: dict,
    expected_oq3: str,
) -> None:
    qc = circuit_factory()
    oq3 = compile_to_oq3(qc, **compile_kwargs)
    assert oq3 == expected_oq3
    result = sim.run(Program(source=oq3), shots=100)
    assert result.result().measurements.shape == (100, qc.num_qubits)


@pytest.mark.parametrize(
    "args_factory,kwargs_factory,exception",
    [
        (lambda: (42,), dict, TypeError),
        (
            lambda: (_bell_circuit(),),
            lambda: {"target": _bell_circuit_target(), "basis_gates": ["h", "cx"]},
            ValueError,
        ),
    ],
    ids=["invalid_input", "conflicting_options"],
)
def test_compile_to_oq3_raises(
    args_factory: Callable[[], tuple],
    kwargs_factory: Callable[[], dict],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        compile_to_oq3(*args_factory(), **kwargs_factory())


def test_compile_to_oq3_output_declaration_survives_and_precedes_verbatim_box() -> None:
    """``output bit[N] c;`` sits outside and above a verbatim box."""
    qc = _output_circuit(("c",), (2,))
    oq3 = compile_to_oq3(qc, verbatim=True, qubit_labels=[0, 1])
    _assert_contents(oq3, ["output bit[2] c;", "#pragma braket verbatim", "box {"], [])
    lines = oq3.split("\n")
    output_idx = next(i for i, ln in enumerate(lines) if ln.startswith("output bit["))
    box_idx = next(i for i, ln in enumerate(lines) if ln == "box {")
    assert output_idx < box_idx


@pytest.mark.parametrize(
    "metadata",
    [{}, {"braket_output_variables": {}}, {"unrelated": "value"}],
    ids=["empty", "empty_output_map", "unrelated_key"],
)
def test_compile_to_oq3_no_output_metadata(metadata: dict) -> None:
    """Without output metadata, no bit declaration is prefixed with `output`."""
    qc = _bell_circuit()
    qc.metadata = metadata
    assert "output bit[" not in compile_to_oq3(qc)
