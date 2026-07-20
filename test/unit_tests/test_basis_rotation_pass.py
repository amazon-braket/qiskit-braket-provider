"""Tests for AddBasisRotationAndMeasurement TransformationPass."""

import numpy as np
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


def _run_pragma_handling_pass(qc: QuantumCircuit) -> QuantumCircuit:
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
    "result_type_cls,observable,target,expected_ops",
    [
        (Expectation, "z", 0, [("measure", [0], [0])]),
        (Expectation, "i", 1, [("measure", [1], [0])]),
        (Expectation, "x", 0, [("h", [0], []), ("measure", [0], [0])]),
        (
            Expectation,
            "y",
            1,
            [("z", [1], []), ("s", [1], []), ("h", [1], []), ("measure", [1], [0])],
        ),
        (Expectation, "h", 0, [("ry", [0], []), ("measure", [0], [0])]),
        (Sample, "x", 0, [("h", [0], []), ("measure", [0], [0])]),
        (Variance, "x", 0, [("h", [0], []), ("measure", [0], [0])]),
    ],
    ids=[
        "z_no_rotation",
        "i_no_rotation",
        "x_h_rotation",
        "y_zsh_rotation",
        "h_ry_rotation",
        "sample_same_as_expectation",
        "variance_same_as_expectation",
    ],
)
def test_single_observable_rotation_and_measurement(
    result_type_cls: type, observable: str, target: int, expected_ops: list
):
    """Each observable/result-type should produce correct rotation gates then measure."""
    pragmas = [result_type_cls(observable=[observable], targets=[target])]
    qc = _circuit_with_result_pragmas(2, pragmas)
    base_count = len(qc.data)

    result = _run_pragma_handling_pass(qc)
    added = _get_ops_after_base(result, base_count)

    assert added == expected_ops


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
        (
            [Probability(targets=[3, 1])],
            4,
            [("measure", [1], [0]), ("measure", [3], [1])],
        ),
        (
            [
                Probability(targets=[0, 1]),
                Expectation(observable=["z"], targets=[0]),
            ],
            2,
            [("measure", [0], [0]), ("measure", [1], [1])],
        ),
        (
            [
                Expectation(observable=["i"], targets=[0]),
                Expectation(observable=["x"], targets=[0]),
            ],
            2,
            [("h", [0], []), ("measure", [0], [0])],
        ),
        (
            [
                Sample(observable=["x"], targets=[0]),
                Expectation(observable=["x"], targets=[0]),
            ],
            2,
            [("h", [0], []), ("measure", [0], [0])],
        ),
        (
            [
                Expectation(observable=["x"], targets=[0]),
                Expectation(observable=["x", "y"], targets=[0, 1]),
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
        (
            [],
            2,
            [],
        ),
        (
            [
                Sample(
                    observable=[[[[0, 0], [1, 0]], [[1, 0], [0, 0]]]],
                    targets=[0],
                ),
                Expectation(
                    observable=[[[[0, 0], [1, 0]], [[1, 0], [0, 0]]]],
                    targets=[0],
                ),
            ],
            2,
            [("unitary", [0], []), ("measure", [0], [0])],
        ),
    ],
    ids=[
        "tensor_product_x_y",
        "probability_explicit_targets",
        "probability_all_qubits",
        "observable_all_qubits",
        "single_observable_broadcast",
        "multiple_result_types",
        "non_monotonic_targets",
        "compatible_z_basis_no_conflict",
        "identity_compatible_with_x",
        "duplicate_observable_deduplicates",
        "partial_overlap_deduplicates_per_qubit",
        "empty_pragmas_no_ops",
        "same_hermitian_deduplicates",
    ],
)
def test_multi_qubit_cases(pragmas: list, num_qubits: int, expected_ops: list):
    """Multi-qubit, broadcast, and multi-result-type cases produce correct operations."""
    qc = _circuit_with_result_pragmas(num_qubits, pragmas)
    base_count = len(qc.data)

    result = _run_pragma_handling_pass(qc)
    added = _get_ops_after_base(result, base_count)

    assert added == expected_ops


def test_hermitian_observable_applies_unitary():
    """Hermitian observable should apply a Unitary that diagonalizes the observable."""
    y_matrix = [[[0.0, 0.0], [0.0, -1.0]], [[0.0, 1.0], [0.0, 0.0]]]
    pragmas = [Expectation(observable=[y_matrix], targets=[0])]
    qc = _circuit_with_result_pragmas(2, pragmas)
    base_count = len(qc.data)

    result = _run_pragma_handling_pass(qc)
    added = _get_ops_after_base(result, base_count)

    assert len(added) == 2
    assert added[0][0] == "unitary"
    assert added[0][1] == [0]
    assert added[1] == ("measure", [0], [0])

    unitary = result.data[base_count].operation.params[0]
    observable = np.array([[0, -1j], [1j, 0]])
    diagonalized = unitary @ observable @ unitary.conj().T
    assert np.allclose(diagonalized, np.diag(np.diag(diagonalized)))

    eigenvalues = result.metadata["braket_pragma_eigenvalues"][0]
    assert np.allclose(sorted(eigenvalues), [-1.0, 1.0])


def test_hermitian_eigenvalues_stored_in_metadata():
    """Eigenvalues from eigh are stored in metadata for downstream consumption."""
    x_matrix = [[[0, 0], [1, 0]], [[1, 0], [0, 0]]]
    z_matrix = [[[1, 0], [0, 0]], [[0, 0], [-1, 0]]]
    pragmas = [
        Expectation(observable=["y"], targets=[1]),
        Expectation(observable=[x_matrix], targets=[0]),
        Expectation(observable=[z_matrix], targets=[2]),
    ]
    qc = _circuit_with_result_pragmas(3, pragmas)

    result = _run_pragma_handling_pass(qc)

    eigen_meta = result.metadata["braket_pragma_eigenvalues"]
    assert 0 not in eigen_meta  # Pauli 'y' has no eigenvalues stored
    assert np.allclose(sorted(eigen_meta[1]), [-1.0, 1.0])  # X eigenvalues
    assert np.allclose(sorted(eigen_meta[2]), [-1.0, 1.0])  # Z eigenvalues


def test_pauli_only_no_eigenvalues_metadata():
    """Pauli-only pragmas should not produce braket_pragma_eigenvalues in metadata."""
    pragmas = [Expectation(observable=["x"], targets=[0])]
    qc = _circuit_with_result_pragmas(2, pragmas)

    result = _run_pragma_handling_pass(qc)

    assert "braket_pragma_eigenvalues" not in result.metadata


def test_hermitian_multi_qubit_endianness():
    """Multi-qubit Hermitian should correct endianness (Braket big-endian → Qiskit little-endian).

    Uses Z⊗X: Z on target[0] (MSB in Braket), X on target[1] (LSB in Braket).
    After endianness correction, the unitary should diagonalize Z⊗X in Qiskit's
    little-endian convention where q[0] = LSB.
    """
    zx = [
        [[0, 0], [1, 0], [0, 0], [0, 0]],
        [[1, 0], [0, 0], [0, 0], [0, 0]],
        [[0, 0], [0, 0], [0, 0], [-1, 0]],
        [[0, 0], [0, 0], [-1, 0], [0, 0]],
    ]
    pragmas = [Expectation(observable=[zx], targets=[0, 1])]
    qc = _circuit_with_result_pragmas(2, pragmas)
    base_count = len(qc.data)

    result = _run_pragma_handling_pass(qc)
    added = _get_ops_after_base(result, base_count)

    assert added[0][0] == "unitary"
    assert added[0][1] == [0, 1]
    assert added[1] == ("measure", [0], [0])
    assert added[2] == ("measure", [1], [1])

    unitary = result.data[base_count].operation.params[0]
    x = np.array([[0, 1], [1, 0]])
    z = np.diag([1, -1])
    observable_qiskit = np.kron(x, z)
    diagonalized = unitary @ observable_qiskit @ unitary.conj().T
    assert np.allclose(diagonalized, np.diag(np.diag(diagonalized)))


def test_mixed_pauli_and_hermitian_tensor_product():
    """Mixed Pauli + Hermitian tensor product correctly accounts for multi-qubit obs."""
    x_matrix = [[[0, 0], [1, 0]], [[1, 0], [0, 0]]]
    pragmas = [Expectation(observable=["z", x_matrix], targets=[0, 1])]
    qc = _circuit_with_result_pragmas(2, pragmas)
    base_count = len(qc.data)

    result = _run_pragma_handling_pass(qc)
    added = _get_ops_after_base(result, base_count)

    assert added[0][0] == "unitary"
    assert added[0][1] == [1]
    assert added[1] == ("measure", [0], [0])
    assert added[2] == ("measure", [1], [1])


def test_fresh_clbits_allocated():
    """Pass always allocates fresh clbits, never reusing existing ones."""
    qc = QuantumCircuit(2, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.metadata = {"braket_result_pragmas": [Probability(targets=[0, 1])]}
    base_count = len(qc.data)

    result = _run_pragma_handling_pass(qc)
    added = _get_ops_after_base(result, base_count)

    assert result.num_clbits == 5
    assert added == [("measure", [0], [3]), ("measure", [1], [4])]
    assert result.metadata["braket_pragma_qubit_to_clbit"] == {0: 3, 1: 4}


@pytest.mark.parametrize(
    "pragmas",
    [
        [StateVector()],
        [DensityMatrix(targets=[0, 1])],
        [Amplitude(states=["00", "11"])],
    ],
    ids=["state_vector", "density_matrix", "amplitude"],
)
def test_basis_invariant_raises_not_implemented(pragmas: list):
    """Basis-invariant types raise NotImplementedError until end-to-end support lands."""
    qc = _circuit_with_result_pragmas(2, pragmas)

    with pytest.raises(NotImplementedError, match="not yet supported end-to-end"):
        _run_pragma_handling_pass(qc)


@pytest.mark.parametrize(
    "pragmas,num_qubits",
    [
        (
            [
                Expectation(observable=["x"], targets=[0]),
                Expectation(observable=["y"], targets=[0]),
            ],
            2,
        ),
        (
            [
                Probability(targets=[0, 1]),
                Expectation(observable=["x"], targets=[0]),
            ],
            2,
        ),
        (
            [
                Expectation(
                    observable=[[[[0, 0], [1, 0]], [[1, 0], [0, 0]]]],
                    targets=[0],
                ),
                Expectation(
                    observable=[[[[1, 0], [0, 0]], [[0, 0], [-1, 0]]]],
                    targets=[0],
                ),
            ],
            2,
        ),
        (
            [
                Expectation(observable=["x"], targets=[0]),
                Expectation(observable=["z", "y"], targets=[0, 1]),
            ],
            2,
        ),
        (
            [
                Expectation(
                    observable=[
                        [
                            [[0, 0], [1, 0], [0, 0], [0, 0]],
                            [[1, 0], [0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0], [-1, 0]],
                            [[0, 0], [0, 0], [-1, 0], [0, 0]],
                        ]
                    ],
                    targets=[0, 1],
                ),
                Expectation(
                    observable=[
                        [
                            [[0, 0], [1, 0], [0, 0], [0, 0]],
                            [[1, 0], [0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0], [-1, 0]],
                            [[0, 0], [0, 0], [-1, 0], [0, 0]],
                        ]
                    ],
                    targets=[0, 2],
                ),
            ],
            3,
        ),
    ],
    ids=[
        "x_vs_y_same_qubit",
        "z_basis_vs_x_same_qubit",
        "different_hermitians_same_qubit",
        "tensor_product_conflicts_with_prior",
        "hermitian_partial_overlap",
    ],
)
def test_conflicting_bases_raises_error(pragmas: list, num_qubits: int):
    """Conflicting measurement bases on the same qubit should raise ValueError."""
    qc = _circuit_with_result_pragmas(num_qubits, pragmas)

    with pytest.raises(ValueError, match=r"(Conflicting measurement bases|partially overlaps)"):
        _run_pragma_handling_pass(qc)


@pytest.mark.parametrize(
    "pragmas,num_qubits,match,error_type",
    [
        (
            [Expectation(observable=["z"], targets=[0])],
            2,
            "already contains measurements",
            ValueError,
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
            ValueError,
        ),
        (
            [Expectation(observable=["x", "y", "z"], targets=[0, 1])],
            2,
            "More observables than target qubits",
            ValueError,
        ),
        (
            [
                Expectation(
                    observable=[
                        [
                            [[0, 0], [0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0], [0, 0]],
                            [[0, 0], [0, 0], [0, 0], [0, 0]],
                        ]
                    ],
                    targets=[0],
                )
            ],
            2,
            "More observables than target qubits",
            ValueError,
        ),
        (
            [Expectation(observable=["x", "y"], targets=[0, 1, 2])],
            3,
            "Fewer observables than target qubits",
            ValueError,
        ),
        (
            ["not_a_result_type"],
            2,
            "Unrecognized result type",
            ValueError,
        ),
        (
            [Expectation(observable=["y"], targets=[0, 0])],
            2,
            "must be unique",
            ValueError,
        ),
    ],
    ids=[
        "double_measurement",
        "hermitian_non_power_of_2",
        "more_observables_than_targets",
        "hermitian_exceeds_targets",
        "fewer_observables_than_targets",
        "unrecognized_result_type",
        "duplicate_targets",
    ],
)
def test_pass_raises_errors(pragmas: list, num_qubits: int, match: str, error_type: type):
    """Pass raises errors for invalid configurations."""
    if match == "already contains measurements":
        qc = QuantumCircuit(num_qubits, 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.metadata = {"braket_result_pragmas": pragmas}
    else:
        qc = _circuit_with_result_pragmas(num_qubits, pragmas)

    with pytest.raises(error_type, match=match):
        _run_pragma_handling_pass(qc)


def test_unknown_observable_raises_error():
    """Unknown observable string should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown observable 'foo'"):
        _rotation_gates_for_observable("foo", 0)


def test_hermitian_invalid_matrix_raises_error():
    """Malformed Hermitian matrix should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid Hermitian matrix format"):
        _rotation_gates_for_hermitian([[("not", "valid")]], [0])
