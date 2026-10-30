"""Tests for the adapter OQ3 output path: ``to_oq3`` and ``compile_to_oq3``."""

import re
import warnings
from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import (
    BoxOp,
    ClassicalRegister,
    Clbit,
    IfElseOp,
    Measure,
    Parameter,
    ParameterVector,
    Qubit,
)
from qiskit.circuit.library import C3SXGate, CXGate, GlobalPhaseGate, HGate, XGate
from qiskit.quantum_info import Statevector
from qiskit.transpiler import PassManager, Target
from qiskit.transpiler.passes import RemoveBarriers

import qiskit_braket_provider
from braket.circuits import Circuit, Gate
from braket.circuits.serialization import (
    IRType,
    OpenQASMSerializationProperties,
    QubitReferenceType,
)
from braket.devices import LocalSimulator
from braket.ir.openqasm import Program
from qiskit_braket_provider.providers.adapter import (
    _compile_to_oq3,
    _has_control_flow,
    compile_to_oq3,
    to_braket,
    to_oq3,
    to_qiskit,
)
from qiskit_braket_provider.providers.gate_mappings import _BRAKET_VERBATIM_BOX_NAME
from qiskit_braket_provider.providers.target import (
    aws_device_to_target,
    native_angle_restrictions,
    native_gate_set,
)

from .mocks import mock_calibrated_rigetti_device, mock_iqm_device


def _to_oq3_from_source(source: str, **kwargs) -> str:
    return to_oq3(to_qiskit(source), **kwargs)


def _loose_qubit_circuit() -> QuantumCircuit:
    qc = QuantumCircuit([Qubit(), Qubit()], [Clbit()])
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(1, 0)
    return qc


def _register_and_loose_qubit_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(QuantumRegister(2, "q"), [Qubit()], ClassicalRegister(1, "c"))
    qc.h(0)
    qc.cx(0, 2)
    qc.measure(2, 0)
    return qc


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


def _ch_and_u_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.ch(0, 1)
    qc.u(0.1, 0.2, 0.3, 0)
    return qc


def _rz_with_circuit_phase() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.global_phase = 0.2
    return qc


def _rz_with_global_phase_gate() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.append(GlobalPhaseGate(0.2), [])
    return qc


def _rz_with_both_global_phases() -> QuantumCircuit:
    qc = _rz_with_global_phase_gate()
    qc.global_phase = 0.2
    return qc


def _rz_with_parameter_phase() -> QuantumCircuit:
    qc = QuantumCircuit(1)
    qc.rz(0.4, 0)
    qc.global_phase = Parameter("p")
    return qc


def _rz_with_global_phase_gate_in_if() -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.rz(0.4, 0)
        qc.append(GlobalPhaseGate(0.2), [])
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


def _reset_circuit() -> QuantumCircuit:
    """Excite qubit 1, measure it, flip it back to |0> on a 1, then re-measure it."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.x(1)
    qc.measure(1, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    qc.measure(1, 1)
    return qc


def _if_with_else_circuit() -> QuantumCircuit:
    """Measure a random qubit 0; reset it to |0> if 1, otherwise flip qubit 1."""
    qc = QuantumCircuit(2, 3)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)) as else_:
        qc.x(0)
    with else_:
        qc.x(1)
    qc.measure(0, 1)
    qc.measure(1, 2)
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


def _rx_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 1)
    circuit.rx(0.3, 0)
    circuit.measure(0, 0)
    return circuit


def _rx_in_if_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2, 1)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.rx(0.3, 1)
    return circuit


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
    ("circuit_factory", "basis_gates", "unsupported"),
    [
        (_rccx_circuit, None, ["rccx"]),
        (_rccx_in_if_else_circuit, None, ["rccx"]),
        (_bell_circuit, ["h"], ["cx"]),
        (_ch_and_u_circuit, ["h"], ["ch", "u"]),
    ],
    ids=[
        "default_basis",
        "nested_in_if_else",
        "explicit_basis",
        "placeholder_names",
    ],
)
def test_to_oq3_raises_on_gates_outside_basis(
    circuit_factory: Callable[[], QuantumCircuit],
    basis_gates: list[str] | None,
    unsupported: list[str],
) -> None:
    """The error names the unsupported gates as Qiskit names them."""
    with pytest.raises(ValueError, match=re.escape(f"Gates {unsupported} are not in the basis")):
        to_oq3(circuit_factory(), basis_gates=basis_gates)


@pytest.mark.parametrize("use_target", [False, True], ids=["braket_device", "target"])
def test_compile_to_oq3_substitutes_rx_for_calibrated_rigetti(use_target: bool) -> None:
    """compile_to_oq3 against a calibrated Rigetti device keeps the substituted rx gates."""
    device = mock_calibrated_rigetti_device()
    if use_target:
        kwargs = {
            "target": aws_device_to_target(device),
            "qubit_labels": sorted(device.topology_graph.nodes),
        }
    else:
        kwargs = {"braket_device": device}

    oq3 = compile_to_oq3(_bell_circuit(), **kwargs)

    _assert_contents(oq3, ["#pragma braket verbatim", "box {", "rx("], ["gate "])
    native_gates = {gate.lower() for gate in device.properties.paradigm.nativeGateSet}
    gate_names = {
        instr.operator.name.lower()
        for instr in Circuit.from_ir(oq3).instructions
        if isinstance(instr.operator, Gate)
    }
    assert gate_names <= native_gates


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


def test_compile_to_oq3_nonstandard_basis_keeps_non_gate_instructions() -> None:
    """A non-standard basis keeps measure, barrier, reset and delay, as the standard basis does."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.barrier()
    circuit.reset(0)
    circuit.delay(100, 1)
    circuit.measure([0, 1], [0, 1])
    standard_basis = ["cz", "rx", "rz"]

    assert compile_to_oq3(circuit, basis_gates=[*standard_basis, "cphaseshift01"]) == (
        compile_to_oq3(circuit, basis_gates=standard_basis)
    )


@pytest.mark.parametrize(
    ("circuit_factory", "expected_block", "expected_outcomes"),
    [
        (_reset_circuit, "if (b[0]) {", {"10"}),
        (_if_with_else_circuit, "} else {", {"100", "001"}),
    ],
    ids=["if", "if_else"],
)
def test_compile_to_oq3_dynamic_circuit_runs_on_local_simulator(
    sim: LocalSimulator,
    circuit_factory: Callable[[], QuantumCircuit],
    expected_block: str,
    expected_outcomes: set[str],
) -> None:
    """Dynamic circuits compile on the default target and run with the expected outcomes."""
    oq3 = compile_to_oq3(circuit_factory())

    assert expected_block in oq3
    counts = sim.run(Program(source=oq3), shots=100).result().measurement_counts
    assert set(counts) == expected_outcomes


def test_compile_to_oq3_default_path_with_iqm_native_gates(sim: LocalSimulator) -> None:
    """IQM circuits with ``if`` compile on the default path, leaving the service pass to run."""
    basis_gates = native_gate_set(mock_iqm_device().properties)

    oq3 = compile_to_oq3(_reset_circuit(), basis_gates=basis_gates, optimization_level=1)

    _assert_contents(oq3, ["if (b[0]) {\nprx("], ["#pragma braket verbatim", "h ", "x "])
    counts = sim.run(Program(source=oq3), shots=100).result().measurement_counts
    assert set(counts) == {"10"}


def test_compile_to_oq3_and_to_oq3_exported_from_package_root() -> None:
    assert qiskit_braket_provider.compile_to_oq3 is compile_to_oq3
    assert qiskit_braket_provider.to_oq3 is to_oq3


@pytest.mark.parametrize(
    ("body", "compile_kwargs", "expected_body"),
    [
        ("h $4;\nb[0] = measure $4;", {}, "h $4;\nb[0] = measure $4;"),
        (
            "h $0;\nx $2;\nb[0] = measure $0;\nif (b[0]) {\nx $2;\n} else {\nh $2;\n}",
            {},
            "h $0;\nx $2;\nb[0] = measure $0;\nif (b[0]) {\nx $2;\n} else {\nh $2;\n}",
        ),
        (
            "h $4;\nb[0] = measure $4;",
            {"verbatim": True},
            "#pragma braket verbatim\nbox {\nh $4;\n}\nb[0] = measure $4;",
        ),
        (
            "rx(1.5707963267948966) $2;\nb[0] = measure $2;",
            {"braket_device": mock_calibrated_rigetti_device(), "verbatim": True},
            "#pragma braket verbatim\nbox {\nrx(1.5707963267948966) $2;\n}\nb[0] = measure $2;",
        ),
    ],
    ids=["gates", "if_else", "verbatim", "verbatim_on_device"],
)
def test_compile_to_oq3_preserves_physical_qubit_labels(
    body: str, compile_kwargs: dict, expected_body: str
) -> None:
    """A program on physical qubits compiles, without warnings, to the same physical qubits."""
    header = "OPENQASM 3.0;\nbit[1] b;\n"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compile_to_oq3(header + body, **compile_kwargs) == header + expected_body


@pytest.mark.parametrize(
    "convert", [compile_to_oq3, _to_oq3_from_source], ids=["compile_to_oq3", "to_oq3"]
)
def test_qubit_labels_ignored_with_warning_for_physical_qubits(convert: Callable) -> None:
    """qubit_labels with a program on physical qubits warns, and the input's qubits are kept."""
    source = "OPENQASM 3.0;\nbit[1] b;\nh $4;\nb[0] = measure $4;"

    with pytest.warns(UserWarning, match="qubit_labels is ignored"):
        assert convert(source, qubit_labels=[0, 1, 2, 3, 9]) == source


def test_to_oq3_keeps_physical_qubits_translated_through_device_labels() -> None:
    """A program translated with physical_qubit_labels keeps its physical qubits."""
    source = "OPENQASM 3.0;\nbit[1] b;\nh $1;\nb[0] = measure $1;"

    assert to_oq3(to_qiskit(source, physical_qubit_labels=list(range(1, 21)))) == source


def test_to_oq3_keeps_hand_built_loose_qubits_virtual() -> None:
    """Loose qubits in a circuit not built by to_qiskit stay virtual, without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        oq3 = to_oq3(_loose_qubit_circuit(), qubit_labels=[10, 11])

    assert "qubit _qubit0;" in oq3
    assert "$" not in oq3


def test_to_oq3_register_with_loose_qubit_and_labels_raises() -> None:
    with pytest.raises(ValueError, match="does not match the circuit's qubit count"):
        to_oq3(_register_and_loose_qubit_circuit(), qubit_labels=[5, 6, 7])


@pytest.mark.parametrize(
    "compile_kwargs",
    [
        {"target": _bell_circuit_target()},
        {"target": aws_device_to_target(mock_calibrated_rigetti_device()), "qubit_labels": (1, 2)},
        {"verbatim": True, "qubit_labels": (0, 1)},
        {"pass_manager": PassManager([RemoveBarriers()]), "qubit_labels": (0, 1)},
        {"pass_manager": PassManager(), "target": _bell_circuit_target()},
    ],
    ids=[
        "target_without_qubits",
        "target_with_qubits",
        "verbatim",
        "pass_manager",
        "empty_pass_manager",
    ],
)
def test_compile_to_oq3_wrap_qubit_specific_targets_only_matches_to_braket(
    compile_kwargs: dict,
) -> None:
    """With wrap_qubit_specific_targets_only, the output is the program to_braket would build."""
    oq3 = _compile_to_oq3(_bell_circuit(), wrap_qubit_specific_targets_only=True, **compile_kwargs)

    assert Circuit.from_ir(oq3) == to_braket(_bell_circuit(), **compile_kwargs)


def test_compile_to_oq3_wrap_qubit_specific_targets_only_keeps_iqm_measurements_in_box() -> None:
    oq3 = _compile_to_oq3(
        _bell_circuit(), wrap_qubit_specific_targets_only=True, **_iqm_device_kwargs()
    )

    assert isinstance(oq3, str)
    assert oq3.endswith("b[1] = measure $2;\n}")


@pytest.mark.parametrize(
    "circuit_factory", [_rx_circuit, _rx_in_if_circuit], ids=["top_level", "in_if"]
)
def test_compile_to_oq3_angle_restrictions_rejects_disallowed_angle(
    circuit_factory: Callable[[], QuantumCircuit],
) -> None:
    circuit = circuit_factory()
    restrictions = native_angle_restrictions(mock_calibrated_rigetti_device().properties)

    with pytest.raises(ValueError, match=r"Angle 0\.3 for rx parameter 0 is not supported"):
        _compile_to_oq3(
            circuit, pass_manager=PassManager([RemoveBarriers()]), angle_restrictions=restrictions
        )


def test_compile_to_oq3_input_names_maps_parameters_to_emitted_inputs() -> None:
    """input_names maps each parameter to its OpenQASM input, including renamed ones."""
    phi = ParameterVector("phi", 2)
    circuit = QuantumCircuit(1)
    circuit.rx(phi[0], 0)
    circuit.ry(phi[1], 0)
    circuit.rz(Parameter("t"), 0)
    circuit.rx(Parameter("alpha"), 0)

    input_names: list[dict[str, str]] = []
    oq3 = _compile_to_oq3(circuit, input_names=input_names)

    assert input_names == [{"alpha": "alpha", "phi[0]": "_phi_0_", "phi[1]": "_phi_1_", "t": "t_0"}]
    for name in input_names[0].values():
        assert f"input float {name};" in oq3


@pytest.mark.parametrize(
    ("gate_name", "add_gate", "expected"),
    [
        ("ch", lambda qc: qc.ch(0, 1), "ctrl @ h q[0], q[1];"),
        ("cs", lambda qc: qc.cs(0, 1), "ctrl @ s q[0], q[1];"),
        ("csdg", lambda qc: qc.csdg(0, 1), "ctrl @ si q[0], q[1];"),
        ("csx", lambda qc: qc.csx(0, 1), "ctrl @ v q[0], q[1];"),
        ("crx", lambda qc: qc.crx(0.5, 0, 1), "ctrl @ rx(0.5) q[0], q[1];"),
        ("cry", lambda qc: qc.cry(0.5, 0, 1), "ctrl @ ry(0.5) q[0], q[1];"),
        ("crz", lambda qc: qc.crz(0.5, 0, 1), "ctrl @ rz(0.5) q[0], q[1];"),
        ("ccz", lambda qc: qc.ccz(0, 1, 2), "ctrl @ cz q[0], q[1], q[2];"),
        (
            "c3sx",
            lambda qc: qc.append(C3SXGate(), [0, 1, 2, 3]),
            "ctrl(3) @ v q[0], q[1], q[2], q[3];",
        ),
        ("u", lambda qc: qc.u(0.1, 0.2, 0.3, 0), "U(0.1, 0.2, 0.3) q[0];"),
    ],
    ids=["ch", "cs", "csdg", "csx", "crx", "cry", "crz", "ccz", "c3sx", "u"],
)
def test_to_oq3_writes_gates_braket_defines(
    sim: LocalSimulator,
    gate_name: str,
    add_gate: Callable[[QuantumCircuit], object],
    expected: str,
) -> None:
    """Controlled gates use Braket's ctrl modifier and u is Braket's U, so the program runs."""
    circuit = QuantumCircuit(4)
    add_gate(circuit)

    oq3 = to_oq3(circuit, basis_gates=[gate_name])

    assert expected in oq3
    sim.run(Program(source=oq3), shots=1).result()


def test_compile_to_oq3_on_local_simulator_target_matches_to_braket() -> None:
    """Compiling against the local simulator's target gives the program to_braket builds."""
    circuit = QuantumCircuit(4)
    circuit.ch(0, 1)
    circuit.ccz(0, 1, 2)
    circuit.append(C3SXGate(), [0, 1, 2, 3])
    circuit.u(0.1, 0.2, 0.3, 0)
    device = LocalSimulator()

    oq3 = compile_to_oq3(circuit, braket_device=device, verbatim=False)

    assert np.allclose(
        Circuit.from_ir(oq3).to_unitary(), to_braket(circuit, braket_device=device).to_unitary()
    )


@pytest.mark.parametrize(
    ("circuit_factory", "expected"),
    [
        (_rz_with_circuit_phase, "gphase(0.2);"),
        (_rz_with_global_phase_gate, "gphase(0.2);"),
        (_rz_with_both_global_phases, "gphase(0.4);"),
        (_rz_with_parameter_phase, "gphase(p);"),
    ],
    ids=["circuit_phase", "gate", "both", "parameter"],
)
def test_to_oq3_writes_global_phase(
    circuit_factory: Callable[[], QuantumCircuit], expected: str
) -> None:
    """Either form of global phase is written as one gphase before the gates."""
    oq3 = to_oq3(circuit_factory(), basis_gates=["rz", "global_phase"])

    lines = oq3.splitlines()
    assert [line for line in lines if "gphase" in line] == [expected]
    assert lines.index(expected) < lines.index("rz(0.4) q[0];")
    assert "global_phase" not in oq3


@pytest.mark.parametrize(
    ("circuit_factory", "basis_gates"),
    [
        (_rz_with_circuit_phase, ["rz"]),
        (_rz_with_circuit_phase, None),
        (_rz_with_global_phase_gate, ["rz"]),
        (_rz_with_global_phase_gate_in_if, ["rz", "if_else", "measure"]),
    ],
    ids=["circuit_phase", "default_basis", "gate", "gate_in_if"],
)
def test_to_oq3_drops_global_phase_outside_basis_with_warning(
    circuit_factory: Callable[[], QuantumCircuit], basis_gates: list[str] | None
) -> None:
    """Either form of global phase is dropped with a warning unless the basis names global_phase."""
    with pytest.warns(UserWarning, match="global phase"):
        oq3 = to_oq3(circuit_factory(), basis_gates=basis_gates)

    assert "gphase" not in oq3


def test_compile_to_oq3_on_local_simulator_keeps_global_phase(sim: LocalSimulator) -> None:
    """The local simulator's state vector matches Qiskit's, global phase included."""
    circuit = QuantumCircuit(1)
    circuit.rz(0.4, 0)
    circuit.global_phase = 0.2

    oq3 = compile_to_oq3(circuit, braket_device=sim, verbatim=False)
    result = sim.run(Program(source=f"{oq3}\n#pragma braket result state_vector"), shots=0).result()

    assert np.allclose(result.values[0], Statevector(circuit).data)
