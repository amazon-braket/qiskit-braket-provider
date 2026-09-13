"""Tests for the adapter OQ3 output path: ``to_oq3`` and ``compile_to_oq3``."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import (
    BoxOp,
    ClassicalRegister,
    Gate,
    IfElseOp,
    Measure,
    Parameter,
)
from qiskit.circuit.library import CXGate, HGate, XGate
from qiskit.qasm3 import QASM3ExporterError
from qiskit.transpiler import Target

from braket.device_schema import DeviceActionType
from braket.devices import LocalSimulator
from braket.ir.openqasm import Program
from qiskit_braket_provider.providers.adapter import (
    _device_executes_control_flow_natively,
    _device_supports_dynamic_circuits,
    _resolve_dynamic_circuits_supported,
    compile_to_oq3,
    to_oq3,
)
from qiskit_braket_provider.providers.gate_mappings import _BRAKET_VERBATIM_BOX_NAME
from qiskit_braket_provider.providers.target import _add_control_flow


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


def _active_reset_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1, 2)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
    qc.measure(0, 1)
    return qc


def _cross_qubit_feedback_circuit() -> QuantumCircuit:
    """The conditioned qubit differs from the measured one."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    qc.measure(1, 1)
    return qc


def _parametric_conditional_circuit() -> QuantumCircuit:
    theta = Parameter("theta")
    qc = QuantumCircuit(1, 2)
    qc.rx(theta, 0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.rx(theta, 0)
    qc.measure(0, 1)
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


def _bell_circuit_target() -> Target:
    target = Target(num_qubits=2)
    target.add_instruction(HGate(), name="h")
    target.add_instruction(CXGate(), name="cx")
    target.add_instruction(Measure(), name="measure")
    return target


def _dynamic_device_target() -> Target:
    """Per-qubit instruction properties, as a QPU target has, plus ``if_else``."""
    target = Target(num_qubits=1)
    target.add_instruction(HGate(), {(0,): None})
    target.add_instruction(XGate(), {(0,): None})
    target.add_instruction(Measure(), {(0,): None})
    target.add_instruction(IfElseOp, name="if_else")
    return target


def _bell_with_verbatim_boxop() -> QuantumCircuit:
    inner = QuantumCircuit(2)
    inner.h(0)
    inner.cx(0, 1)
    outer = QuantumCircuit(2, 2)
    outer.append(BoxOp(inner, label=_BRAKET_VERBATIM_BOX_NAME), [0, 1])
    outer.measure([0, 1], [0, 1])
    return outer


_ACTIVE_RESET_OQ3 = (
    "OPENQASM 3.0;\n"
    "bit[2] b;\n"
    "qubit[1] q;\n"
    "h q[0];\n"
    "b[0] = measure q[0];\n"
    "if (b[0]) {\n"
    "x q[0];\n"
    "}\n"
    "b[1] = measure q[0];"
)


def test_to_oq3_auto_basis_gates() -> None:
    """Omitting ``basis_gates`` assumes Braket's gate set, so no definitions appear."""
    oq3 = to_oq3(_bell_circuit())
    _assert_contents(oq3, ["h ", "cnot "], ["gate "])


def test_compile_to_oq3_list_input() -> None:
    results = compile_to_oq3([_bell_circuit(), _bell_circuit()])
    assert isinstance(results, list)
    assert len(results) == 2
    assert all("OPENQASM 3.0;" in r for r in results)


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
        (lambda: ("not a circuit",), dict, TypeError),
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


def test_compile_to_oq3_output_declaration_survives_verbatim() -> None:
    """A verbatim-wrapped compile still emits ``output bit[N] c;`` outside the box.

    The rest of the metadata → declaration flow is unit-covered by
    ``test_consolidate_clbits`` and ``test_normalize_formatting``; this case
    exercises the interaction with :class:`WrapInVerbatimBox` which lives only
    in the ``compile_to_oq3`` pipeline.
    """
    qc = _output_circuit(("c",), (2,))
    oq3 = compile_to_oq3(qc, verbatim=True, qubit_labels=[0, 1])
    _assert_contents(oq3, ["output bit[2] c;", "#pragma braket verbatim", "box {"], [])


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
def test_compile_to_oq3_no_output_metadata(metadata: dict) -> None:
    """Without output metadata, no bit declaration is prefixed with `output`."""
    qc = _bell_circuit()
    qc.metadata = metadata
    assert "output bit[" not in compile_to_oq3(qc)


def _mock_device_with_operations(supported_operations: list[str] | None) -> MagicMock:
    """Build a mock Braket ``Device``.

    If ``supported_operations`` is ``None``, the mock advertises no OpenQASM
    action; otherwise it advertises the given ``supportedOperations``.
    """
    device = MagicMock()
    if supported_operations is None:
        device.properties.action = {}
    else:
        action = MagicMock()
        action.supportedOperations = supported_operations
        device.properties.action = {DeviceActionType.OPENQASM: action}
    return device


def _target_with(*operations: str) -> Target:
    t = Target(num_qubits=2)
    t.add_instruction(HGate(), name="h")
    for op in operations:
        if op == "if_else":
            t.add_instruction(IfElseOp, name="if_else")
    return t


@pytest.mark.parametrize(
    "supported_operations,expected",
    [
        (["h", "cnot"], False),
        (["h", "measure_ff"], True),
        (["h", "cc_prx"], True),
        (["h", "if"], True),
        (["MEASURE_FF", "H"], True),
        (None, False),
    ],
    ids=["no_dynamic", "measure_ff", "cc_prx", "if", "case_insensitive", "no_openqasm_action"],
)
def test_device_supports_dynamic_circuits(
    supported_operations: list[str] | None, expected: bool
) -> None:
    assert (
        _device_supports_dynamic_circuits(_mock_device_with_operations(supported_operations))
        is expected
    )


@pytest.mark.parametrize(
    "explicit,device,target,basis_gates,expected",
    [
        (None, None, None, None, False),
        (True, None, None, None, True),
        (False, None, None, None, False),
        (None, _mock_device_with_operations(["h", "measure_ff"]), None, None, True),
        (None, _mock_device_with_operations(["h", "cnot"]), None, None, False),
        (None, None, _target_with("if_else"), None, True),
        (None, None, _target_with(), None, False),
        (None, None, None, ["h", "cx", "if_else"], True),
        (None, None, None, ["h", "cx"], False),
        (True, _mock_device_with_operations(["h"]), None, None, True),
    ],
    ids=[
        "all_none",
        "explicit_true",
        "explicit_false",
        "device_dynamic",
        "device_static",
        "target_if_else",
        "target_static",
        "basis_gates_if_else",
        "basis_gates_static",
        "explicit_overrides_device",
    ],
)
def test_resolve_dynamic_circuits_supported(
    explicit: bool | None,
    device: MagicMock | None,
    target: Target | None,
    basis_gates: list[str] | None,
    expected: bool,
) -> None:
    assert _resolve_dynamic_circuits_supported(explicit, device, target, basis_gates) is expected


@pytest.mark.parametrize(
    "dynamic_circuits_supported,expected_oq3",
    [
        (
            True,
            (
                "OPENQASM 3.0;\n"
                "bit[2] b;\n"
                "qubit[2] q;\n"
                "h q[0];\n"
                "b[0] = measure q[0];\n"
                "cnot q[0], q[1];\n"
                "b[1] = measure q[1];"
            ),
        ),
        (
            False,
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
    ],
    ids=["dynamic_preserves_placement", "static_moves_measurements_to_end"],
)
def test_compile_to_oq3_respects_dynamic_circuits_supported(
    dynamic_circuits_supported: bool, expected_oq3: str
) -> None:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)
    qc.measure(1, 1)
    assert compile_to_oq3(qc, dynamic_circuits_supported=dynamic_circuits_supported) == expected_oq3


def test_compile_to_oq3_dynamic_circuit() -> None:
    oq3 = compile_to_oq3(_active_reset_circuit(), dynamic_circuits_supported=True)
    assert oq3 == _ACTIVE_RESET_OQ3


def test_compile_to_oq3_dynamic_circuit_on_device_target() -> None:
    oq3 = compile_to_oq3(
        _active_reset_circuit(),
        target=_dynamic_device_target(),
        qubit_labels=[7],
        dynamic_circuits_supported=True,
    )
    _assert_contents(
        oq3,
        ["h $7;", "b[0] = measure $7;", "if (b[0]) {", "x $7;", "b[1] = measure $7;"],
        ["#pragma braket verbatim", "box {", "qubit["],
    )


def test_compile_to_oq3_control_flow_keeps_measurement_before_branch() -> None:
    """Holds without dynamic_circuits_supported, which the circuit itself implies."""
    oq3 = compile_to_oq3(_active_reset_circuit(), basis_gates=["h", "x"])
    assert oq3 == _ACTIVE_RESET_OQ3


def test_compile_to_oq3_dynamic_circuit_accepted_by_braket_simulator(sim: LocalSimulator) -> None:
    oq3 = compile_to_oq3(_active_reset_circuit(), dynamic_circuits_supported=True)
    counts = sim.run(Program(source=oq3), shots=100).result().measurement_counts
    assert sum(counts.values()) == 100
    # b[1] is measured after the conditional x, so the qubit is always back in |0>
    assert all(key[1] == "0" for key in counts)


def test_compile_to_oq3_cross_qubit_feedback() -> None:
    oq3 = compile_to_oq3(
        _cross_qubit_feedback_circuit(), basis_gates=["h", "x"], qubit_labels=[1, 2]
    )
    _assert_contents(
        oq3,
        ["h $1;", "b[0] = measure $1;", "if (b[0]) {", "x $2;", "b[1] = measure $2;"],
        ["qubit["],
    )


def test_compile_to_oq3_cross_qubit_feedback_accepted_by_braket_simulator(
    sim: LocalSimulator,
) -> None:
    oq3 = compile_to_oq3(_cross_qubit_feedback_circuit(), basis_gates=["h", "x"])
    counts = sim.run(Program(source=oq3), shots=100).result().measurement_counts
    assert sum(counts.values()) == 100
    # qubit 1 is flipped exactly when qubit 0 measured 1, so both bits always agree
    assert all(key[0] == key[1] for key in counts)


def test_compile_to_oq3_parametric_dynamic_circuit() -> None:
    oq3 = compile_to_oq3(_parametric_conditional_circuit(), basis_gates=["rx"])
    _assert_contents(
        oq3,
        ["input float theta;", "rx(theta) q[0];", "if (b[0]) {"],
        ["float[64]", "input float theta_0;"],
    )
    # the body's gate is the same parameter, declared once
    assert oq3.count("input float theta;") == 1
    assert oq3.count("rx(theta) q[0];") == 2


@pytest.mark.parametrize(
    "supported_operations,expected_ops",
    [(["h", "cnot"], set()), (["h", "IF"], {"if_else"})],
    ids=["no_if", "if"],
)
def test_add_control_flow(supported_operations: list[str], expected_ops: set[str]) -> None:
    target = Target(num_qubits=1)
    action = MagicMock()
    action.supportedOperations = supported_operations
    _add_control_flow(target, action)
    assert set(target.operation_names) == expected_ops


def test_to_oq3_gate_only_inside_if_body_needs_no_definition() -> None:
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.sx(0)
    _assert_contents(to_oq3(qc, dynamic_circuits_supported=True), ["v q[0];"], ["gate "])


def test_to_oq3_gate_outside_braket_gate_set_raises() -> None:
    """Emitted bare for the service to reject before; now it fails here."""
    qc = QuantumCircuit(1, 1)
    qc.append(Gate("mygate", 1, []), [0])
    with pytest.raises(QASM3ExporterError, match="mygate"):
        to_oq3(qc)


@pytest.mark.parametrize(
    "native_gate_set,expected",
    [
        (["cz", "prx", "cc_prx", "measure_ff", "barrier"], False),
        (["cz", "prx", "if"], True),
        (["cz", "prx", "IF"], True),
        ([], False),
    ],
    ids=["feedback_primitives_only", "native_if", "case_insensitive", "empty"],
)
def test_device_executes_control_flow_natively(native_gate_set: list[str], expected: bool) -> None:
    device = MagicMock()
    device.properties.paradigm.nativeGateSet = native_gate_set
    assert _device_executes_control_flow_natively(device) is expected
