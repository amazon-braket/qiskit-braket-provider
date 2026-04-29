import numpy as np
import pytest
from qiskit.circuit.library import standard_gates as qiskit_gates
from qiskit.transpiler import Target

from braket.circuits import Circuit
from qiskit_braket_provider.providers.adapter import to_braket


def _make_target(num_qubits: int) -> Target:
    target = Target(num_qubits=num_qubits)
    one_q = {(i,): None for i in range(num_qubits)}
    two_q = {(i, j): None for i in range(num_qubits) for j in range(num_qubits) if i != j}
    target.add_instruction(qiskit_gates.HGate(), one_q)
    target.add_instruction(qiskit_gates.XGate(), one_q)
    target.add_instruction(qiskit_gates.YGate(), one_q)
    target.add_instruction(qiskit_gates.ZGate(), one_q)
    target.add_instruction(qiskit_gates.SGate(), one_q)
    target.add_instruction(qiskit_gates.TGate(), one_q)
    target.add_instruction(qiskit_gates.RXGate(0.0), one_q)
    target.add_instruction(qiskit_gates.RYGate(0.0), one_q)
    target.add_instruction(qiskit_gates.RZGate(0.0), one_q)
    target.add_instruction(qiskit_gates.CXGate(), two_q)
    target.add_instruction(qiskit_gates.CZGate(), two_q)
    target.add_instruction(qiskit_gates.SwapGate(), two_q)
    if num_qubits >= 3:
        target.add_instruction(
            qiskit_gates.CCXGate(),
            {
                (i, j, k): None
                for i in range(num_qubits)
                for j in range(num_qubits)
                for k in range(num_qubits)
                if len({i, j, k}) == 3
            },
        )
    return target


def _unitaries_equal(u1: np.ndarray, u2: np.ndarray) -> bool:
    product = u1 @ np.conj(u2.T)
    phase = product[0, 0]
    return np.allclose(product, phase * np.eye(product.shape[0]))


def _extract_gate_lines(qasm: str) -> list[str]:
    skip = {"OPENQASM", "bit", "qubit", "//", "#pragma", "box", "}"}
    return [
        line.strip()
        for line in qasm.strip().splitlines()
        if line.strip()
        and line.strip() != "{"
        and not any(line.strip().startswith(s) for s in skip)
        and "= measure" not in line
    ]


def _get_braket_gate_names(braket_circuit: Circuit) -> list[str]:
    directives = {"StartVerbatimBox", "EndVerbatimBox"}
    return [
        instr.operator.name
        for instr in braket_circuit.instructions
        if instr.operator.name not in directives
    ]


def _assert_verbatim_roundtrip(qasm: str, **kwargs) -> str:
    bc = to_braket(qasm, **kwargs)
    ir_source = bc.to_ir(ir_type="OPENQASM").source

    assert "#pragma braket verbatim" in ir_source
    assert "$" in ir_source
    assert "qubit[" not in ir_source

    reference = Circuit.from_ir(qasm)
    assert _unitaries_equal(bc.to_unitary(), reference.to_unitary())

    return ir_source


def test_single_h_with_target() -> None:
    _assert_verbatim_roundtrip("OPENQASM 3.0;\nh $0;\n", target=_make_target(1))


def test_bell_state_with_target() -> None:
    _assert_verbatim_roundtrip("OPENQASM 3.0;\nh $0;\ncnot $0, $1;\n", target=_make_target(2))


def test_three_qubit_toffoli_with_target() -> None:
    _assert_verbatim_roundtrip(
        "OPENQASM 3.0;\nh $0;\nh $1;\nccnot $0, $1, $2;\n", target=_make_target(3)
    )


def test_rotation_gates_with_target() -> None:
    _assert_verbatim_roundtrip(
        "OPENQASM 3.0;\nrx(1.5707963267948966) $0;\nry(0.7853981633974483) $1;\ncnot $0, $1;\n",
        target=_make_target(2),
    )


def test_swap_gate_with_target() -> None:
    _assert_verbatim_roundtrip("OPENQASM 3.0;\nswap $0, $1;\n", target=_make_target(2))


def test_multi_gate_sequence_with_target() -> None:
    _assert_verbatim_roundtrip(
        "OPENQASM 3.0;\nx $0;\nh $0;\ncnot $0, $1;\ny $1;\n", target=_make_target(2)
    )


def test_has_measurements_with_target() -> None:
    result = _assert_verbatim_roundtrip(
        "OPENQASM 3.0;\nh $0;\ncnot $0, $1;\n", target=_make_target(2)
    )
    assert "measure" in result


@pytest.mark.parametrize(
    "qasm, num_qubits",
    [
        ("OPENQASM 3.0;\nh $0;\n", 1),
        ("OPENQASM 3.0;\ncnot $0, $1;\n", 2),
        ("OPENQASM 3.0;\nccnot $0, $1, $2;\n", 3),
    ],
    ids=["1_qubit", "2_qubits", "3_qubits"],
)
def test_qubit_count_preserved(qasm: str, num_qubits: int) -> None:
    bc = to_braket(qasm, target=_make_target(num_qubits))
    assert bc.qubit_count == num_qubits


def test_simple_circuit_verbatim() -> None:
    qasm = "OPENQASM 3.0;\nh $0;\ncnot $0, $1;\n"
    result = _assert_verbatim_roundtrip(qasm, verbatim=True)
    assert _extract_gate_lines(result) == _extract_gate_lines(qasm)


def test_preserves_gate_order_verbatim() -> None:
    qasm = "OPENQASM 3.0;\nx $0;\nh $0;\ncnot $0, $1;\ny $1;\n"
    result = _assert_verbatim_roundtrip(qasm, verbatim=True)
    assert _extract_gate_lines(result) == _extract_gate_lines(qasm)


def test_three_qubits_verbatim() -> None:
    qasm = "OPENQASM 3.0;\nh $0;\nh $1;\nccnot $0, $1, $2;\n"
    result = _assert_verbatim_roundtrip(qasm, verbatim=True)
    assert _extract_gate_lines(result) == _extract_gate_lines(qasm)


def test_rotation_gates_verbatim() -> None:
    qasm = "OPENQASM 3.0;\nrx(1.5707963267948966) $0;\ncnot $0, $1;\n"
    result = _assert_verbatim_roundtrip(qasm, verbatim=True)
    assert _extract_gate_lines(result) == _extract_gate_lines(qasm)


def test_single_verbatim_box_with_target() -> None:
    _assert_verbatim_roundtrip(
        "OPENQASM 3.0;\n#pragma braket verbatim\nbox {\n    h $0;\n    cnot $0, $1;\n}\n",
        target=_make_target(2),
    )


def test_mixed_verbatim_and_regular_with_target() -> None:
    _assert_verbatim_roundtrip(
        "OPENQASM 3.0;\n"
        "x $0;\n"
        "#pragma braket verbatim\n"
        "box {\n"
        "    h $0;\n"
        "    cnot $0, $1;\n"
        "}\n"
        "y $1;\n",
        target=_make_target(2),
    )


def test_multiple_verbatim_boxes_with_target() -> None:
    _assert_verbatim_roundtrip(
        "OPENQASM 3.0;\n"
        "#pragma braket verbatim\n"
        "box {\n"
        "    h $0;\n"
        "}\n"
        "x $1;\n"
        "#pragma braket verbatim\n"
        "box {\n"
        "    cnot $0, $1;\n"
        "}\n",
        target=_make_target(2),
    )


def test_verbatim_box_with_verbatim_true() -> None:
    _assert_verbatim_roundtrip(
        "OPENQASM 3.0;\n"
        "x $0;\n"
        "#pragma braket verbatim\n"
        "box {\n"
        "    h $0;\n"
        "    cnot $0, $1;\n"
        "}\n"
        "y $1;\n",
        verbatim=True,
    )


def test_h_decomposed_to_rx_rz() -> None:
    target = Target(num_qubits=1)
    one_q = {(0,): None}
    target.add_instruction(qiskit_gates.RXGate(0.0), one_q)
    target.add_instruction(qiskit_gates.RZGate(0.0), one_q)
    bc = to_braket("OPENQASM 3.0;\nh $0;\n", target=target)
    _assert_verbatim_roundtrip("OPENQASM 3.0;\nh $0;\n", target=target)
    assert set(_get_braket_gate_names(bc)).issubset({"Rx", "Rz"})


def test_bell_state_decomposed_to_rx_rz_cx() -> None:
    target = Target(num_qubits=2)
    one_q = {(0,): None, (1,): None}
    two_q = {(0, 1): None, (1, 0): None}
    target.add_instruction(qiskit_gates.RXGate(0.0), one_q)
    target.add_instruction(qiskit_gates.RZGate(0.0), one_q)
    target.add_instruction(qiskit_gates.CXGate(), two_q)
    qasm = "OPENQASM 3.0;\nh $0;\ncnot $0, $1;\n"
    bc = to_braket(qasm, target=target)
    _assert_verbatim_roundtrip(qasm, target=target)
    assert set(_get_braket_gate_names(bc)).issubset({"Rx", "Rz", "CNot"})


def test_swap_decomposed() -> None:
    target = Target(num_qubits=2)
    one_q = {(0,): None, (1,): None}
    two_q = {(0, 1): None, (1, 0): None}
    target.add_instruction(qiskit_gates.RXGate(0.0), one_q)
    target.add_instruction(qiskit_gates.RZGate(0.0), one_q)
    target.add_instruction(qiskit_gates.CXGate(), two_q)
    qasm = "OPENQASM 3.0;\nswap $0, $1;\n"
    bc = to_braket(qasm, target=target)
    _assert_verbatim_roundtrip(qasm, target=target)
    assert set(_get_braket_gate_names(bc)).issubset({"Rx", "Rz", "CNot"})


def test_toffoli_decomposed() -> None:
    target = Target(num_qubits=3)
    one_q = {(i,): None for i in range(3)}
    two_q = {(i, j): None for i in range(3) for j in range(3) if i != j}
    target.add_instruction(qiskit_gates.RXGate(0.0), one_q)
    target.add_instruction(qiskit_gates.RZGate(0.0), one_q)
    target.add_instruction(qiskit_gates.CXGate(), two_q)
    qasm = "OPENQASM 3.0;\nh $0;\nh $1;\nccnot $0, $1, $2;\n"
    bc = to_braket(qasm, target=target)
    _assert_verbatim_roundtrip(qasm, target=target)
    assert set(_get_braket_gate_names(bc)).issubset({"Rx", "Rz", "CNot"})
