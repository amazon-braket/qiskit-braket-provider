"""Transpiler pass to add basis rotation gates and measurements for Braket result type pragmas."""

import math

import numpy as np
from qiskit.circuit import Clbit, Gate, Measure
from qiskit.circuit.library import HGate, RYGate, SGate, UnitaryGate, ZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from braket.ir.jaqcd import (
    Expectation,
    Probability,
    Sample,
    Variance,
)
from qiskit_braket_provider.providers.gate_mappings import (
    _BASIS_INVARIANT_RESULT_TYPES,
    _OBSERVABLE_RESULT_TYPES,
    _Z_BASIS_RESULT_TYPES,
)

RotationOp = tuple[Gate, int | list[int]]

_OBSERVABLE_TO_BASIS = {
    "z": "z",
    "i": "z",
    "x": "x",
    "y": "y",
    "h": "h",
}


def _reverse_endianness(matrix: np.ndarray) -> np.ndarray:
    """Reverse qubit endianness of a matrix (Braket big-endian to Qiskit little-endian).

    For single-qubit matrices this is a no-op. For multi-qubit matrices, the tensor
    factor ordering is reversed so q[0] becomes LSB (Qiskit convention) instead of
    MSB (Braket convention).

    Args:
        matrix: Square matrix of dimension 2^n x 2^n.

    Returns:
        Matrix with reversed qubit ordering.
    """
    n_q = int(np.log2(matrix.shape[0]))
    if n_q <= 1:
        return matrix
    return np.transpose(
        matrix.reshape([2] * n_q * 2),
        list(range(n_q))[::-1] + list(range(n_q, 2 * n_q))[::-1],
    ).reshape((2**n_q, 2**n_q))


def _rotation_gates_for_observable(observable: str, target: int) -> list[RotationOp]:
    """Return (gate, qubit) pairs for rotating from the given observable basis to Z basis.

    Args:
        observable: Single-character observable name (x, y, z, h, i).
        target: Qubit index to apply the rotation to.

    Returns:
        List of (gate, target) tuples. Empty list for Z/I (already in Z basis).

    Raises:
        ValueError: If the observable name is not recognized.
    """
    match observable.lower():
        case "z" | "i":
            return []
        case "x":
            return [(HGate(), target)]
        case "y":
            return [(ZGate(), target), (SGate(), target), (HGate(), target)]
        case "h":
            return [(RYGate(-math.pi / 4), target)]
        case _:
            raise ValueError(f"Unknown observable '{observable}' in result type pragma.")


def _rotation_gates_for_hermitian(matrix: list, targets: list[int]) -> list[RotationOp]:
    """Compute rotation gates for a Hermitian observable from its eigenvectors.

    The resulting unitary is endianness-corrected from Braket's big-endian convention
    to Qiskit's little-endian convention before being wrapped in a UnitaryGate.

    Args:
        matrix: Nested list representing the Hermitian matrix that defines the observable,
            where each element is a [real, imag] pair.
        targets: Qubit indices the unitary should be applied to.

    Returns:
        List of (gate, targets) tuples.

    Raises:
        ValueError: If the matrix cannot be parsed or is not a valid Hermitian matrix.
    """
    try:
        np_matrix = np.array([[complex(c[0], c[1]) for c in row] for row in matrix])
    except (IndexError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid Hermitian matrix format in result type pragma: {e}") from e
    np_matrix = _reverse_endianness(np_matrix)
    _eigenvalues, eigenvectors = np.linalg.eigh(np_matrix)
    unitary = eigenvectors.conj().T
    return [(UnitaryGate(unitary), targets)]


def _qubits_for_z_basis_type(result: Probability, num_qubits: int) -> set[int]:
    """Return the set of qubit indices to measure for a Z-basis result type.

    Args:
        result: A Probability result type.
        num_qubits: Total number of qubits in the circuit.

    Returns:
        Set of qubit indices to measure. No rotation is needed for Z-basis types.
    """
    targets = result.targets
    if targets is not None:
        return set(targets)
    return set(range(num_qubits))


def _basis_for_observable(obs: str) -> str:
    """Return the measurement basis key for a string observable.

    Args:
        obs: Observable name string (x, y, z, h, i).

    Returns:
        Basis string: "z", "x", "y", or "h".
    """
    return _OBSERVABLE_TO_BASIS.get(obs.lower(), obs.lower())


def _check_basis_conflicts(
    qubit_bases: dict[int, str], new_qubits: set[int], new_basis: str
) -> None:
    """Check and record basis assignments, raising on conflicts.

    Z-basis (from Probability or z/i observables) is compatible with other Z-basis
    assignments but conflicts with any other basis on the same qubit.

    Args:
        qubit_bases: Mutable dict mapping qubit index to its committed basis.
        new_qubits: Set of qubit indices being assigned.
        new_basis: The basis being assigned to these qubits.

    Raises:
        ValueError: If a qubit already has a different non-compatible basis assigned.
    """
    for qubit in new_qubits:
        existing = qubit_bases.get(qubit)
        if existing is not None and existing != new_basis:
            raise ValueError(
                f"Conflicting measurement bases on qubit {qubit}: "
                f"'{existing}' and '{new_basis}' cannot be measured simultaneously."
            )
        qubit_bases[qubit] = new_basis


def _plan_for_observable_type(
    result: Expectation | Sample | Variance, num_qubits: int
) -> tuple[list[RotationOp], dict[int, str]]:
    """Compute the rotation/measurement plan for an observable result type.

    Args:
        result: An Expectation, Sample, or Variance result type.
        num_qubits: Total number of qubits in the circuit.

    Returns:
        Tuple of (rotation_ops, qubit_bases) where qubit_bases maps qubit index
        to its measurement basis string.

    Raises:
        ValueError: If a hermitian matrix size is not a power of 2, or if
            there are more observables than target qubits.
    """
    observable = result.observable
    targets = result.targets if result.targets is not None else list(range(num_qubits))

    rotation_ops: list[RotationOp] = []
    qubit_bases: dict[int, str] = {}

    if len(observable) == 1 and isinstance(observable[0], str):
        basis = _basis_for_observable(observable[0])
        for target in targets:
            rotation_ops.extend(_rotation_gates_for_observable(observable[0], target))
            qubit_bases[target] = basis
    else:
        obs_idx = 0
        for obs in observable:
            if isinstance(obs, str):
                if obs_idx >= len(targets):
                    raise ValueError("More observables than target qubits in result type pragma.")
                target = targets[obs_idx]
                rotation_ops.extend(_rotation_gates_for_observable(obs, target))
                qubit_bases[target] = _basis_for_observable(obs)
                obs_idx += 1
            elif isinstance(obs, list):
                if len(obs) == 0 or (len(obs) & (len(obs) - 1)) != 0:
                    raise ValueError(f"Hermitian matrix size {len(obs)} is not a power of 2.")
                num_qubits_for_obs = int(math.log2(len(obs)))
                if obs_idx + num_qubits_for_obs > len(targets):
                    raise ValueError("More observables than target qubits in result type pragma.")
                obs_targets = targets[obs_idx : obs_idx + num_qubits_for_obs]
                rotation_ops.extend(_rotation_gates_for_hermitian(obs, obs_targets))
                for t in obs_targets:
                    qubit_bases[t] = "hermitian"
                obs_idx += num_qubits_for_obs

        if obs_idx != len(targets):
            raise ValueError("Fewer observables than target qubits in result type pragma.")

    return rotation_ops, qubit_bases


class AddBasisRotationAndMeasurement(TransformationPass):
    """Append basis rotation gates and measurements for result type pragmas.

    Reads ``braket_result_pragmas`` from the circuit metadata and appends
    the appropriate rotation gates to change from the observable's eigenbasis
    to the computational (Z) basis, followed by measurements on all targeted qubits.

    For Z-basis types (probability), only measurements are added. For observable
    types (expectation, sample, variance), rotation gates are added before measurements.

    Measurements are always written to freshly allocated classical bits appended to
    the circuit. Existing classical bits are never reused. Measurements are added in
    sorted qubit order: qubit N maps to classical bit index based on its position in
    the sorted set of all measured qubits. For example, if targets are {3, 1},
    measurements will be q[1]→c[new+0], q[3]→c[new+1].

    This pass should run before transpilation so that added rotation gates are
    compiled to the device's native gate set and qubit layout is applied correctly.

    Raises:
        ValueError: If the circuit already contains measurement operations.
        ValueError: If two result pragmas require different measurement bases on
            the same qubit.
        NotImplementedError: If basis-invariant result types (state_vector,
            density_matrix, amplitude) are present, as these are not yet supported
            end-to-end.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Add basis rotation gates and measurements to the DAG.

        Args:
            dag: The DAG circuit to transform.

        Returns:
            The modified DAG with rotation gates and measurements appended.

        Raises:
            ValueError: If the circuit already contains measurement operations, or
                if conflicting measurement bases are requested on the same qubit.
            NotImplementedError: If basis-invariant result types are present.
        """
        metadata = dag.metadata or {}
        result_pragmas = metadata.get("braket_result_pragmas", [])

        if not result_pragmas:
            return dag

        existing_measures = dag.count_ops().get("measure", 0)
        if existing_measures > 0:
            raise ValueError(
                "Circuit already contains measurements. Cannot add result type "
                "measurements on top of existing ones."
            )

        num_qubits = dag.num_qubits()
        all_rotation_ops: list[RotationOp] = []
        all_qubit_bases: dict[int, str] = {}

        for result in result_pragmas:
            if isinstance(result, _BASIS_INVARIANT_RESULT_TYPES):
                raise NotImplementedError(
                    f"{type(result).__name__} result types are not yet supported "
                    "end-to-end. Support will be added in a future release."
                )

            if isinstance(result, _Z_BASIS_RESULT_TYPES):
                qubits = _qubits_for_z_basis_type(result, num_qubits)
                _check_basis_conflicts(all_qubit_bases, qubits, "z")
                continue

            if isinstance(result, _OBSERVABLE_RESULT_TYPES):
                rotation_ops, qubit_bases = _plan_for_observable_type(result, num_qubits)
                for qubit, basis in qubit_bases.items():
                    _check_basis_conflicts(all_qubit_bases, {qubit}, basis)
                all_rotation_ops.extend(rotation_ops)
                continue

            raise ValueError(f"Unrecognized result type: {type(result).__name__}")

        pragma_clbits = [Clbit() for _ in range(len(all_qubit_bases))]
        dag.add_clbits(pragma_clbits)

        for gate, target in all_rotation_ops:
            if isinstance(target, list):
                dag.apply_operation_back(gate, [dag.qubits[t] for t in target])
            else:
                dag.apply_operation_back(gate, [dag.qubits[target]])

        for idx, qubit in enumerate(sorted(all_qubit_bases)):
            dag.apply_operation_back(Measure(), [dag.qubits[qubit]], [pragma_clbits[idx]])

        dag.metadata["braket_pragma_qubit_to_clbit"] = {
            qubit: dag.clbits.index(pragma_clbits[idx])
            for idx, qubit in enumerate(sorted(all_qubit_bases))
        }

        return dag
