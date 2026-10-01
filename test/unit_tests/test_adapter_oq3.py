"""Tests for the adapter OQ3 output path: ``to_oq3`` and ``compile_to_oq3``."""

import warnings
from collections.abc import Callable
from unittest.mock import MagicMock

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

from braket.device_schema import DeviceActionType
from braket.devices import LocalSimulator
from braket.ir.openqasm import Program
from qiskit_braket_provider.providers.adapter import (
    _collect_basis_gates,
    _device_supports_dynamic_circuits,
    _has_control_flow,
    _resolve_preserve_measurement_order,
    compile_to_oq3,
    to_oq3,
)
from qiskit_braket_provider.providers.gate_mappings import _BRAKET_VERBATIM_BOX_NAME


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
    """Omitting ``basis_gates`` triggers ``_collect_basis_gates`` on the circuit."""
    oq3 = to_oq3(_bell_circuit())
    _assert_contents(oq3, ["h ", "cnot "], ["gate "])


def test_collect_basis_gates_recurses_into_control_flow_bodies() -> None:
    """Regression: gates inside ``.blocks``-carrying ops (e.g. IfElseOp) are collected."""
    true_body = QuantumCircuit(1, 1)
    true_body.x(0)

    outer = QuantumCircuit(1, 1)
    outer.h(0)
    outer.measure(0, 0)
    outer.append(IfElseOp((outer.clbits[0], 1), true_body, None), [0], [0])

    gates = _collect_basis_gates(outer.data)
    assert "x" in gates
    assert "h" in gates
    assert "if_else" not in gates


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


@pytest.mark.parametrize(
    "circuit_factory,expected",
    [(_if_else_circuit, True), (_bell_circuit, False)],
    ids=["if_else_circuit", "plain_circuit"],
)
def test_has_control_flow(circuit_factory: Callable[[], QuantumCircuit], expected: bool) -> None:
    assert _has_control_flow(circuit_factory()) is expected


def test_compile_to_oq3_skips_verbatim_on_native_path_with_control_flow() -> None:
    """Native path skips the verbatim wrap for control-flow circuits and warns."""
    qc = _if_else_circuit()

    with pytest.warns(UserWarning, match="control-flow"):
        oq3 = compile_to_oq3(qc, target=_if_else_target())

    assert "#pragma braket verbatim" not in oq3
    assert "box {" not in oq3


def test_compile_to_oq3_wraps_verbatim_when_explicit_even_with_control_flow() -> None:
    """``verbatim=True`` is honored for control-flow circuits; no warning fires."""
    qc = _if_else_circuit()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        oq3 = compile_to_oq3(qc, verbatim=True, preserve_measurement_order=True)

    assert "#pragma braket verbatim" in oq3
    assert "box {" in oq3


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
def test_resolve_preserve_measurement_order(
    explicit: bool | None,
    device: MagicMock | None,
    target: Target | None,
    basis_gates: list[str] | None,
    expected: bool,
) -> None:
    assert _resolve_preserve_measurement_order(explicit, device, target, basis_gates) is expected


@pytest.mark.parametrize(
    "preserve_measurement_order,expected_oq3",
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
    ids=["preserve_keeps_placement", "reorder_moves_measurements_to_end"],
)
def test_compile_to_oq3_respects_preserve_measurement_order(
    preserve_measurement_order: bool, expected_oq3: str
) -> None:
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)
    qc.measure(1, 1)
    assert compile_to_oq3(qc, preserve_measurement_order=preserve_measurement_order) == expected_oq3
