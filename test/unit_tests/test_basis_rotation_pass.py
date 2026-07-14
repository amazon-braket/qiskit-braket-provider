"""Tests for AddBasisRotationAndMeasurement TransformationPass."""

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from braket.ir.jaqcd import (
    Amplitude,
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)
from qiskit_braket_provider.providers.passes import AddBasisRotationAndMeasurement
from qiskit_braket_provider.providers.passes.basis_rotation_pass import (
    _rotation_gates_for_hermitian,
    _rotation_gates_for_observable,
)


def _circuit_with_result_pragmas(num_qubits: int, pragmas: list) -> QuantumCircuit:
    """Helper to create a circuit with result pragma metadata."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    if num_qubits > 1:
        qc.cx(0, 1)
    qc.metadata = {"braket_result_pragmas": pragmas}
    return qc


def _run_pass(qc: QuantumCircuit) -> QuantumCircuit:
    """Run the AddBasisRotationAndMeasurement pass on a circuit."""
    return PassManager([AddBasisRotationAndMeasurement()]).run(qc)


def _get_ops_after_base(
    result: QuantumCircuit, base_count: int
) -> list[tuple[str, list[int], list[int]]]:
    """Extract (name, qubit_indices, clbit_indices) for operations added by the pass.

    Assumes PassManager preserves instruction order, so the first `base_count`
    instructions are the original circuit and everything after is pass-added.
    """
    added = []
    for instr in result.data[base_count:]:
        qubits = [result.qubits.index(q) for q in instr.qubits]
        clbits = [result.clbits.index(c) for c in instr.clbits]
        added.append((instr.operation.name, qubits, clbits))
    return added


@pytest.mark.parametrize(
    "observable,target,expected_ops",
    [
        ("z", 0, [("measure", [0], [0])]),
        ("i", 1, [("measure", [1], [0])]),
        ("x", 0, [("h", [0], []), ("measure", [0], [0])]),
        (
            "y",
            1,
            [("z", [1], []), ("s", [1], []), ("h", [1], []), ("measure", [1], [0])],
        ),
        ("h", 0, [("ry", [0], []), ("measure", [0], [0])]),
    ],
    ids=["z_no_rotation", "i_no_rotation", "x_h_rotation", "y_zsh_rotation", "h_ry_rotation"],
)
def test_single_observable_rotation_and_measurement(
    observable: str, target: int, expected_ops: list
):
    """Each observable should produce correct rotation gates on the target qubit then measure."""
    pragmas = [Expectation(observable=[observable], targets=[target])]
    qc = _circuit_with_result_pragmas(2, pragmas)

    result = _run_pass(qc)
    added = _get_ops_after_base(result, 2)

    assert added == expected_ops


@pytest.mark.parametrize(
    "result_type_cls",
    [Expectation, Sample, Variance],
    ids=["expectation", "sample", "variance"],
)
def test_observable_types_all_produce_same_rotation(result_type_cls: type):
    """Expectation, Sample, Variance all produce identical rotation + measurement."""
    pragmas = [result_type_cls(observable=["x"], targets=[0])]
    qc = _circuit_with_result_pragmas(2, pragmas)

    result = _run_pass(qc)
    added = _get_ops_after_base(result, 2)

    assert added == [("h", [0], []), ("measure", [0], [0])]


@pytest.mark.parametrize(
    "pragmas,num_qubits,expected_ops",
    [
        (
            [Expectation(observable=["x", "y"], targets=[0, 1])],
            2,
            [
                ("h", [0], []),
                ("z", [1], []),
                ("s", [1], []),
                ("h", [1], []),
                ("measure", [0], [0]),
                ("measure", [1], [1]),
            ],
        ),
        (
            [Probability(targets=[0, 1])],
            2,
            [("measure", [0], [0]), ("measure", [1], [1])],
        ),
        (
            [Probability(targets=None)],
            3,
            [("measure", [0], [0]), ("measure", [1], [1]), ("measure", [2], [2])],
        ),
        (
            [Expectation(observable=["x"], targets=None)],
            3,
            [
                ("h", [0], []),
                ("h", [1], []),
                ("h", [2], []),
                ("measure", [0], [0]),
                ("measure", [1], [1]),
                ("measure", [2], [2]),
            ],
        ),
        (
            [Expectation(observable=["x"], targets=[0, 1, 2])],
            3,
            [
                ("h", [0], []),
                ("h", [1], []),
                ("h", [2], []),
                ("measure", [0], [0]),
                ("measure", [1], [1]),
                ("measure", [2], [2]),
            ],
        ),
        (
            [
                Expectation(observable=["x"], targets=[0]),
                Variance(observable=["y"], targets=[1]),
            ],
            2,
            [
                ("h", [0], []),
                ("z", [1], []),
                ("s", [1], []),
                ("h", [1], []),
                ("measure", [0], [0]),
                ("measure", [1], [1]),
            ],
        ),
    ],
    ids=[
        "tensor_product_x_y",
        "probability_explicit_targets",
        "probability_all_qubits",
        "observable_all_qubits",
        "single_observable_broadcast",
        "multiple_result_types",
    ],
)
def test_multi_qubit_cases(pragmas: list, num_qubits: int, expected_ops: list):
    """Multi-qubit, broadcast, and multi-result-type cases produce correct operations."""
    qc = _circuit_with_result_pragmas(num_qubits, pragmas)

    result = _run_pass(qc)
    added = _get_ops_after_base(result, 2)

    assert added == expected_ops


def test_hermitian_observable_applies_unitary():
    """Hermitian observable should apply a Unitary gate on the correct qubit."""
    y_matrix = [[[0.0, 0.0], [0.0, -1.0]], [[0.0, 1.0], [0.0, 0.0]]]
    pragmas = [Expectation(observable=[y_matrix], targets=[0])]
    qc = _circuit_with_result_pragmas(2, pragmas)

    result = _run_pass(qc)
    added = _get_ops_after_base(result, 2)

    assert len(added) == 2
    assert added[0][0] == "unitary"
    assert added[0][1] == [0]
    assert added[1] == ("measure", [0], [0])


@pytest.mark.parametrize(
    "pragmas",
    [
        [StateVector()],
        [DensityMatrix(targets=[0, 1])],
        [Amplitude(states=["00", "11"])],
        [],
        None,
    ],
    ids=["state_vector", "density_matrix", "amplitude", "empty_pragmas", "no_metadata"],
)
def test_no_ops_added(pragmas: list):
    """Basis-invariant types and empty pragmas should not modify the circuit."""
    qc = _circuit_with_result_pragmas(2, pragmas)

    result = _run_pass(qc)
    assert _get_ops_after_base(result, 2) == []


def test_unknown_observable_raises_error():
    """Unknown observable string should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown observable 'foo'"):
        _rotation_gates_for_observable("foo", 0)


def test_hermitian_invalid_matrix_raises_error():
    """Malformed Hermitian matrix should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid Hermitian matrix format"):
        _rotation_gates_for_hermitian([[("not", "valid")]], [0])


@pytest.mark.parametrize(
    "pragmas,num_qubits,match",
    [
        (
            [Expectation(observable=["z"], targets=[0])],
            2,
            "already contains measurements",
        ),
        (
            [
                Expectation(
                    observable=[
                        [
                            [[0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0]],
                        ]
                    ],
                    targets=[0],
                )
            ],
            2,
            "not a power of 2",
        ),
        (
            [Expectation(observable=["x", "y", "z"], targets=[0, 1])],
            2,
            "More observables than target qubits",
        ),
        (
            [Expectation(observable=["x", "y"], targets=[0, 1, 2])],
            3,
            "Fewer observables than target qubits",
        ),
        (
            ["not_a_result_type"],
            2,
            "Unrecognized result type",
        ),
    ],
    ids=[
        "double_measurement",
        "hermitian_non_power_of_2",
        "more_observables_than_targets",
        "fewer_observables_than_targets",
        "unrecognized_result_type",
    ],
)
def test_pass_raises_errors(pragmas: list, num_qubits: int, match: str):
    """Pass raises ValueError for invalid configurations."""
    if match == "already contains measurements":
        qc = QuantumCircuit(num_qubits, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.metadata = {"braket_result_pragmas": pragmas}
    else:
        qc = _circuit_with_result_pragmas(num_qubits, pragmas)

    pm = PassManager([AddBasisRotationAndMeasurement()])
    with pytest.raises(ValueError, match=match):
        pm.run(qc)
