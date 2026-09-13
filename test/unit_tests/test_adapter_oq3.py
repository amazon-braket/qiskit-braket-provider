"""Tests for the adapter OQ3 output path: ``to_oq3`` and ``compile_to_oq3``."""

from collections.abc import Callable
from unittest.mock import MagicMock, Mock

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

from braket.aws import AwsDevice
from braket.device_schema import DeviceActionType
from braket.devices import LocalSimulator
from braket.ir.openqasm import Program
from test.unit_tests.mocks import (
    MOCK_IQM_GATE_MODEL_QPU_CAPABILITIES,
    MOCK_IQM_TOPOLOGY_GRAPH,
)
from qiskit_braket_provider.providers.adapter import (
    _device_executes_control_flow_natively,
    _device_supports_dynamic_circuits,
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


def _if_else_circuit() -> QuantumCircuit:
    """Both branches act, so the qubit ends in |0> either way."""
    qc = QuantumCircuit(1, 2)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)) as else_:
        qc.x(0)
    with else_:
        qc.z(0)
    qc.measure(0, 1)
    return qc


def _nested_conditional_circuit() -> QuantumCircuit:
    """An inner branch conditioned on a measurement taken inside the outer branch."""
    qc = QuantumCircuit(2, 3)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.h(1)
        qc.measure(1, 1)
        with qc.if_test((qc.clbits[1], 1)):
            qc.x(0)
    qc.measure(0, 2)
    return qc


def _teleportation_circuit() -> QuantumCircuit:
    """Teleport |1> from q0 to q2 with the usual X and Z corrections."""
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    with qc.if_test((qc.clbits[1], 1)):
        qc.x(2)
    with qc.if_test((qc.clbits[0], 1)):
        qc.z(2)
    qc.measure(2, 2)
    return qc


def _measure_all_circuit() -> QuantumCircuit:
    """The measure_all() idiom, which adds its own "meas" register and a barrier."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
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


def _iqm_device() -> Mock:
    """An IQM QPU whose published operations include the if statement."""
    device = Mock(spec=AwsDevice)
    device.properties = MOCK_IQM_GATE_MODEL_QPU_CAPABILITIES
    device.gate_calibrations = None
    device.type = "QPU"
    device.topology_graph = MOCK_IQM_TOPOLOGY_GRAPH
    return device


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


@pytest.mark.parametrize(
    "supported_operations,expected",
    [(["h", "if", "measure"], True), (["h", "measure"], False), (None, False)],
    ids=["supports_if", "no_if", "no_openqasm_action"],
)
def test_device_supports_dynamic_circuits(
    supported_operations: list[str] | None, expected: bool
) -> None:
    device = MagicMock()
    action = MagicMock()
    action.supportedOperations = supported_operations
    device.properties.action = (
        {} if supported_operations is None else {DeviceActionType.OPENQASM: action}
    )
    assert _device_supports_dynamic_circuits(device) is expected


def test_to_oq3_control_flow() -> None:
    """A branch in the circuit is enough to stop the reordering, with no device given."""
    assert to_oq3(_active_reset_circuit(), basis_gates=["h", "x"]) == _ACTIVE_RESET_OQ3.replace(
        "$0", "q[0]"
    )


def test_compile_to_oq3_moves_mid_circuit_measurement() -> None:
    """Nothing reads the measured bit, so the measurement is moved to the end."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.cx(0, 1)
    qc.measure(1, 1)
    assert compile_to_oq3(qc) == (
        "OPENQASM 3.0;\n"
        "bit[2] b;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "b[0] = measure q[0];\n"
        "b[1] = measure q[1];"
    )


def test_compile_to_oq3_dynamic_circuit() -> None:
    oq3 = compile_to_oq3(_active_reset_circuit())
    assert oq3 == _ACTIVE_RESET_OQ3


def test_compile_to_oq3_dynamic_target() -> None:
    oq3 = compile_to_oq3(
        _active_reset_circuit(),
        target=_dynamic_device_target(),
        qubit_labels=[7],
    )
    _assert_contents(
        oq3,
        ["h $7;", "b[0] = measure $7;", "if (b[0]) {", "x $7;", "b[1] = measure $7;"],
        ["#pragma braket verbatim", "box {", "qubit["],
    )


def test_compile_to_oq3_measurement_precedes_branch() -> None:
    """Holds without preserve_measurement_order, which the circuit itself implies."""
    oq3 = compile_to_oq3(_active_reset_circuit(), basis_gates=["h", "x"])
    assert oq3 == _ACTIVE_RESET_OQ3


def test_dynamic_circuit_on_simulator(sim: LocalSimulator) -> None:
    oq3 = compile_to_oq3(_active_reset_circuit())
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


def test_cross_qubit_feedback_on_simulator(
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


def test_to_oq3_gate_inside_if_body() -> None:
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.sx(0)
    _assert_contents(to_oq3(qc), ["v q[0];"], ["gate "])


def test_to_oq3_gate_outside_braket_gate_set_raises() -> None:
    """A gate Braket does not define is rejected here rather than emitted bare."""
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


def test_compile_to_oq3_if_else() -> None:
    assert compile_to_oq3(_if_else_circuit()) == (
        "OPENQASM 3.0;\n"
        "bit[2] b;\n"
        "qubit[1] q;\n"
        "h q[0];\n"
        "b[0] = measure q[0];\n"
        "if (b[0]) {\n"
        "x q[0];\n"
        "} else {\n"
        "z q[0];\n"
        "}\n"
        "b[1] = measure q[0];"
    )


def test_compile_to_oq3_nested_conditional() -> None:
    oq3 = compile_to_oq3(_nested_conditional_circuit())
    assert oq3.splitlines()[3:] == [
        "h q[0];",
        "b[0] = measure q[0];",
        "if (b[0]) {",
        "h q[1];",
        "b[1] = measure q[1];",
        "if (b[1]) {",
        "x q[0];",
        "}",
        "}",
        "b[2] = measure q[0];",
    ]


@pytest.mark.parametrize(
    "build_circuit,assert_counts",
    [
        (_active_reset_circuit, lambda counts: all(key[1] == "0" for key in counts)),
        (_if_else_circuit, lambda counts: all(key[1] == "0" for key in counts)),
        (
            _nested_conditional_circuit,
            lambda counts: all(key in {"000", "101", "110"} for key in counts),
        ),
        (_teleportation_circuit, lambda counts: all(key[2] == "1" for key in counts)),
    ],
    ids=["active_reset", "if_else", "nested_conditional", "teleportation"],
)
def test_iqm_program_on_simulator(
    sim: LocalSimulator,
    build_circuit: Callable[[], QuantumCircuit],
    assert_counts: Callable[[dict], bool],
) -> None:
    """Run the IQM-native program to check the lowering to prx and cz preserved the outcomes.

    The simulator accepts these gates, so it can execute a program compiled for the QPU. This
    is what backs the expected strings above, which on their own only pin the output down.
    """
    oq3 = compile_to_oq3(build_circuit(), braket_device=_iqm_device())
    counts = sim.run(Program(source=oq3), shots=200).result().measurement_counts
    assert sum(counts.values()) == 200
    assert assert_counts(counts)


def test_compile_to_oq3_measure_all() -> None:
    assert compile_to_oq3(_measure_all_circuit()) == (
        "OPENQASM 3.0;\n"
        "bit[2] b;\n"
        "qubit[2] q;\n"
        "h q[0];\n"
        "cnot q[0], q[1];\n"
        "barrier q[0], q[1];\n"
        "b[0] = measure q[0];\n"
        "b[1] = measure q[1];"
    )


def test_compile_to_oq3_batch() -> None:
    """A conditional circuit keeps its measurement placement; its plain sibling is normalized."""
    plain = QuantumCircuit(2, 2)
    plain.h(0)
    plain.measure(0, 0)
    plain.cx(0, 1)
    plain.measure(1, 1)

    plain_oq3, conditional_oq3 = compile_to_oq3([plain, _active_reset_circuit()])
    assert plain_oq3.splitlines()[-2:] == ["b[0] = measure q[0];", "b[1] = measure q[1];"]
    conditional_lines = conditional_oq3.splitlines()
    assert conditional_lines.index("b[0] = measure q[0];") < conditional_lines.index("if (b[0]) {")


@pytest.mark.parametrize(
    "build_circuit,expected_oq3",
    [
        (
            _active_reset_circuit,
            """OPENQASM 3.0;
bit[2] b;
prx(1.5707963267948966, 1.5707963267948966) $1;
prx(3.141592653589793, 0.0) $1;
b[0] = measure $1;
if (b[0]) {
prx(3.141592653589793, 0.0) $1;
}
b[1] = measure $1;""",
        ),
        (
            _if_else_circuit,
            """OPENQASM 3.0;
bit[2] b;
prx(1.5707963267948966, 1.5707963267948966) $1;
prx(3.141592653589793, 0.0) $1;
b[0] = measure $1;
if (b[0]) {
prx(3.141592653589793, 0.0) $1;
} else {
prx(3.141592653589793, 0.0) $1;
prx(3.141592653589793, 1.5707963267948966) $1;
}
b[1] = measure $1;""",
        ),
        (
            _teleportation_circuit,
            """OPENQASM 3.0;
bit[3] b;
prx(3.141592653589793, 0.0) $1;
prx(1.5707963267948966, 1.5707963267948966) $2;
prx(3.141592653589793, 0.0) $2;
prx(1.5707963267948966, 0.0) $2;
prx(1.5707963267948966, 1.5707963267948966) $3;
prx(3.141592653589793, 0.0) $3;
prx(1.5707963267948966, 0.0) $4;
prx(1.5707963267948966, 0.0) $5;
cz $5, $2;
prx(1.5707963267948966, 0.0) $2;
prx(1.5707963267948966, 0.0) $5;
cz $5, $2;
prx(1.5707963267948966, 0.0) $2;
prx(1.5707963267948966, 0.0) $5;
cz $5, $2;
prx(1.5707963267948966, 0.0) $5;
cz $4, $5;
prx(1.5707963267948966, 0.0) $4;
prx(1.5707963267948966, 0.0) $5;
cz $4, $5;
prx(1.5707963267948966, 0.0) $4;
prx(1.5707963267948966, 0.0) $5;
cz $4, $5;
cz $4, $3;
prx(1.5707963267948966, 1.5707963267948966) $3;
prx(3.141592653589793, 0.0) $3;
prx(1.5707963267948966, 1.5707963267948966) $4;
prx(3.141592653589793, 0.0) $4;
cz $1, $4;
prx(1.5707963267948966, 1.5707963267948966) $1;
prx(3.141592653589793, 0.0) $1;
prx(1.5707963267948966, 1.5707963267948966) $4;
prx(3.141592653589793, 0.0) $4;
b[0] = measure $1;
b[1] = measure $4;
if (b[1]) {
prx(3.141592653589793, 0.0) $3;
}
if (b[0]) {
prx(3.141592653589793, 0.0) $3;
prx(3.141592653589793, 1.5707963267948966) $3;
}
b[2] = measure $3;""",
        ),
    ],
    ids=["active_reset", "if_else", "teleportation"],
)
def test_compile_to_oq3_on_iqm_device(
    build_circuit: Callable[[], QuantumCircuit], expected_oq3: str
) -> None:
    """IQM native gates, physical qubits, and a branch needing translation, so no verbatim box.

    The seed is needed because layout and routing are otherwise chosen afresh per process,
    changing which physical qubits the circuit lands on.
    """
    oq3 = compile_to_oq3(build_circuit(), braket_device=_iqm_device(), seed_transpiler=42)
    assert oq3 == expected_oq3
