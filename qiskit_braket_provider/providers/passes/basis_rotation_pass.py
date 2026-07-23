"""Transpiler pass to add basis rotation gates and measurements for Braket result type pragmas."""

import math
from collections.abc import Hashable

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
    _IDENTITY_BASIS,
    _OBSERVABLE_RESULT_TYPES,
    _OBSERVABLE_TO_BASIS,
    _Z_BASIS_RESULT_TYPES,
    _reverse_endianness,
)

RotationOp = tuple[Gate, int | list[int]]


def _hermitian_key(matrix: list) -> Hashable:
    """Create a hashable key from a Hermitian matrix for deduplication.

    Args:
        matrix: Nested list representing the Hermitian matrix in [real, imag] format.

    Returns:
        A hashable tuple of complex elements for identity comparison.
    """
    return tuple(complex(c[0], c[1]) for row in matrix for c in row)


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


def _rotation_gates_for_hermitian(
    matrix: list, targets: list[int]
) -> tuple[list[RotationOp], np.ndarray]:
    """Compute rotation gates for a Hermitian observable from its eigenvectors.

    The resulting unitary is endianness-corrected from Braket's big-endian convention
    to Qiskit's little-endian convention before being wrapped in a UnitaryGate.

    Args:
        matrix: Nested list representing the Hermitian matrix that defines the observable,
            where each element is a [real, imag] pair.
        targets: Qubit indices the unitary should be applied to.

    Returns:
        Tuple of (rotation_ops, eigenvalues) where eigenvalues are in ascending order
        as returned by np.linalg.eigh. Measurement bit i corresponds to eigenvalue[i].

    Raises:
        ValueError: If the matrix cannot be parsed or is not a valid Hermitian matrix.
    """
    try:
        np_matrix = np.array([[complex(c[0], c[1]) for c in row] for row in matrix])
    except (IndexError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid Hermitian matrix format in result type pragma: {e}") from e
    np_matrix = _reverse_endianness(np_matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(np_matrix)
    unitary = eigenvectors.conj().T
    return [(UnitaryGate(unitary), targets)], eigenvalues


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


def _check_basis_conflict(
    qubit_bases: dict[int, Hashable], qubit: int, basis_key: Hashable
) -> bool:
    """Check and record a single qubit's basis assignment, raising on conflicts.

    Same basis on the same qubit is deduplicated (returns False). Different basis
    on the same qubit raises ValueError. New qubit assignment returns True.
    Identity ("i") is compatible with any basis and does not override it.

    Args:
        qubit_bases: Mutable dict mapping qubit index to its committed basis key.
        qubit: The qubit index being assigned.
        basis_key: Hashable key identifying the observable basis. "i" means identity
            (compatible with anything). For Pauli observables this is the string from
            _OBSERVABLE_TO_BASIS. For Hermitian observables this is a tuple of the
            matrix elements.

    Returns:
        True if the qubit was newly committed, False if it already had the same basis
        or the new basis is identity.

    Raises:
        ValueError: If the qubit already has a different non-identity basis assigned.
    """
    if qubit not in qubit_bases:
        qubit_bases[qubit] = basis_key
        return True

    existing = qubit_bases[qubit]

    # Identity is diagonal in every basis (never conflicts); same basis deduplicates
    if basis_key == _IDENTITY_BASIS or existing == basis_key:
        return False

    # Upgrade from identity to a real basis
    if existing == _IDENTITY_BASIS:
        qubit_bases[qubit] = basis_key
        return True

    raise ValueError(
        f"Conflicting measurement bases on qubit {qubit}: "
        f"cannot measure simultaneously with different observables."
    )


def _plan_for_observable_type(
    result: Expectation | Sample | Variance, num_qubits: int
) -> tuple[list[RotationOp], dict[int, Hashable], dict[tuple[int, ...], np.ndarray]]:
    """Compute the rotation/measurement plan for an observable result type.

    Args:
        result: An Expectation, Sample, or Variance result type.
        num_qubits: Total number of qubits in the circuit.

    Returns:
        Tuple of (rotation_ops, qubit_bases, eigenvalues_by_qubits) where:
        - qubit_bases maps qubit index to a hashable basis key
        - eigenvalues_by_qubits maps qubit tuples to their Hermitian eigenvalue
          arrays (empty dict for Pauli-only observables)

    Raises:
        ValueError: If a hermitian matrix size is not a power of 2, or if
            there are more observables than target qubits.
    """
    observable = result.observable
    targets = result.targets if result.targets is not None else list(range(num_qubits))

    if len(set(targets)) != len(targets):
        raise ValueError(f"All targets in a result-type pragma must be unique: {targets}")

    rotation_ops: list[RotationOp] = []
    qubit_bases: dict[int, Hashable] = {}
    eigenvalues_by_qubits: dict[tuple[int, ...], np.ndarray] = {}

    if len(observable) == 1 and isinstance(observable[0], str):
        basis = _OBSERVABLE_TO_BASIS.get(observable[0].lower(), observable[0].lower())
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
                qubit_bases[target] = _OBSERVABLE_TO_BASIS.get(obs.lower(), obs.lower())
                obs_idx += 1
            elif isinstance(obs, list):
                if len(obs) == 0 or (len(obs) & (len(obs) - 1)) != 0:
                    raise ValueError(f"Hermitian matrix size {len(obs)} is not a power of 2.")
                num_qubits_for_obs = int(math.log2(len(obs)))
                if obs_idx + num_qubits_for_obs > len(targets):
                    raise ValueError("More observables than target qubits in result type pragma.")
                obs_targets = targets[obs_idx : obs_idx + num_qubits_for_obs]
                hermitian_ops, eigenvalues = _rotation_gates_for_hermitian(obs, obs_targets)
                rotation_ops.extend(hermitian_ops)
                eigenvalues_by_qubits[tuple(obs_targets)] = eigenvalues
                h_key = _hermitian_key(obs)
                for t in obs_targets:
                    qubit_bases[t] = h_key
                obs_idx += num_qubits_for_obs

        if obs_idx != len(targets):
            raise ValueError("Fewer observables than target qubits in result type pragma.")

    return rotation_ops, qubit_bases, eigenvalues_by_qubits


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
    the sorted set of all measured qubits.

    This pass should run before transpilation so that added rotation gates are
    compiled to the device's native gate set and qubit layout is applied correctly.

    If multiple result pragmas target the same qubit with the same observable, rotation
    gates are applied only once (deduplicated). If different observables target the same
    qubit, the pass raises ValueError since they cannot be measured simultaneously.

    Raises:
        ValueError: If the circuit already contains measurement operations.
        ValueError: If two result pragmas require different measurement bases on
            the same qubit (non-commuting observables).
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
        all_qubit_bases: dict[int, Hashable] = {}
        all_eigenvalues: dict[tuple[int, ...], np.ndarray] = {}

        for result in result_pragmas:
            if isinstance(result, _BASIS_INVARIANT_RESULT_TYPES):
                continue

            if isinstance(result, _Z_BASIS_RESULT_TYPES):
                qubits = _qubits_for_z_basis_type(result, num_qubits)
                for qubit in qubits:
                    _check_basis_conflict(all_qubit_bases, qubit, "z")
                continue

            if isinstance(result, _OBSERVABLE_RESULT_TYPES):
                rotation_ops, qubit_bases, eigenvalues_by_qubits = _plan_for_observable_type(
                    result, num_qubits
                )
                for qubit_group, eigenvalues in eigenvalues_by_qubits.items():
                    if qubit_group not in all_eigenvalues:
                        all_eigenvalues[qubit_group] = eigenvalues
                new_qubits: set[int] = set()
                for qubit, basis in qubit_bases.items():
                    if _check_basis_conflict(all_qubit_bases, qubit, basis):
                        new_qubits.add(qubit)
                # Only add rotation ops targeting qubits not yet rotated
                for gate, target in rotation_ops:
                    gate_qubits = set(target) if isinstance(target, list) else {target}
                    if gate_qubits <= new_qubits:
                        all_rotation_ops.append((gate, target))
                    elif gate_qubits & new_qubits:
                        raise ValueError(
                            "Multi-qubit observable partially overlaps with previously "
                            "committed qubits. Cannot apply rotation to a subset of "
                            "an entangled observable's targets."
                        )
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

        if all_eigenvalues:
            dag.metadata["braket_pragma_eigenvalues"] = all_eigenvalues

        return dag
