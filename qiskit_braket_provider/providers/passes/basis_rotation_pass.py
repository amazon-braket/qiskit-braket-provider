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
    _eigenvalues, eigenvectors = np.linalg.eigh(np_matrix)
    unitary = eigenvectors.conj().T
    return [(UnitaryGate(unitary), targets)]


def _plan_for_z_basis_type(result: Probability, num_qubits: int) -> set[int]:
    """Compute the qubits to measure for a Z-basis result type.

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


def _plan_for_observable_type(
    result: Expectation | Sample | Variance, num_qubits: int
) -> tuple[list[RotationOp], set[int]]:
    """Compute the rotation/measurement plan for an observable result type.

    Args:
        result: An Expectation, Sample, or Variance result type.
        num_qubits: Total number of qubits in the circuit.

    Returns:
        Tuple of (rotation_ops, qubits_to_measure).

    Raises:
        ValueError: If a hermitian matrix size is not a power of 2, or if
            there are more observables than target qubits.
    """
    observable = result.observable
    targets = result.targets if result.targets is not None else list(range(num_qubits))

    rotation_ops: list[RotationOp] = []
    # All targeted qubits are measured, including identity observables.
    # This matches Braket SDK behavior where measurement results are returned
    # for every qubit in the target list regardless of the observable type.
    qubits_to_measure: set[int] = set()

    if len(observable) == 1 and isinstance(observable[0], str):
        for target in targets:
            rotation_ops.extend(_rotation_gates_for_observable(observable[0], target))
            qubits_to_measure.add(target)
    else:
        obs_idx = 0
        for obs in observable:
            if isinstance(obs, str):
                if obs_idx >= len(targets):
                    raise ValueError("More observables than target qubits in result type pragma.")
                target = targets[obs_idx]
                rotation_ops.extend(_rotation_gates_for_observable(obs, target))
                qubits_to_measure.add(target)
                obs_idx += 1
            elif isinstance(obs, list):
                if len(obs) == 0 or (len(obs) & (len(obs) - 1)) != 0:
                    raise ValueError(f"Hermitian matrix size {len(obs)} is not a power of 2.")
                num_qubits_for_obs = int(math.log2(len(obs)))
                if obs_idx + num_qubits_for_obs > len(targets):
                    raise ValueError("More observables than target qubits in result type pragma.")
                obs_targets = targets[obs_idx : obs_idx + num_qubits_for_obs]
                rotation_ops.extend(_rotation_gates_for_hermitian(obs, obs_targets))
                qubits_to_measure.update(obs_targets)
                obs_idx += num_qubits_for_obs

        if obs_idx != len(targets):
            raise ValueError("Fewer observables than target qubits in result type pragma.")

    return rotation_ops, qubits_to_measure


class AddBasisRotationAndMeasurement(TransformationPass):
    """Append basis rotation gates and measurements for result type pragmas.

    Reads ``braket_result_pragmas`` from the circuit metadata and appends
    the appropriate rotation gates to change from the observable's eigenbasis
    to the computational (Z) basis, followed by measurements on all targeted qubits.

    For basis-invariant result types (state_vector, density_matrix, amplitude),
    no gates or measurements are added. For Z-basis types (probability), only
    measurements are added. For observable types (expectation, sample, variance),
    rotation gates are added before measurements.

    This pass should run before transpilation so that added rotation gates are
    compiled to the device's native gate set and qubit layout is applied correctly.

    If the circuit already contains measurements, this pass will raise a ValueError
    to prevent double-measurement.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Add basis rotation gates and measurements to the DAG.

        Args:
            dag: The DAG circuit to transform.

        Returns:
            The modified DAG with rotation gates and measurements appended.

        Raises:
            ValueError: If the circuit already contains measurement operations.
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
        all_qubits_to_measure: set[int] = set()

        for result in result_pragmas:
            if isinstance(result, _BASIS_INVARIANT_RESULT_TYPES):
                continue

            if isinstance(result, _Z_BASIS_RESULT_TYPES):
                qubits = _plan_for_z_basis_type(result, num_qubits)
                all_qubits_to_measure.update(qubits)
                continue

            if isinstance(result, _OBSERVABLE_RESULT_TYPES):
                rotation_ops, qubits = _plan_for_observable_type(result, num_qubits)
                all_rotation_ops.extend(rotation_ops)
                all_qubits_to_measure.update(qubits)
                continue

            raise ValueError(f"Unrecognized result type: {type(result).__name__}")

        if not all_qubits_to_measure and not all_rotation_ops:
            return dag

        num_clbits_needed = len(all_qubits_to_measure)
        clbits_to_add = num_clbits_needed - dag.num_clbits()
        if clbits_to_add > 0:
            for _ in range(clbits_to_add):
                dag.add_clbits([Clbit()])

        for gate, target in all_rotation_ops:
            if isinstance(target, list):
                dag.apply_operation_back(gate, [dag.qubits[t] for t in target])
            else:
                dag.apply_operation_back(gate, [dag.qubits[target]])

        # Measurements are added in sorted qubit order, mapping qubit N to
        # classical bit index based on position in the sorted set. E.g., if
        # targets are {3, 1}, measurements will be: q[1]→c[0], q[3]→c[1].
        for idx, qubit in enumerate(sorted(all_qubits_to_measure)):
            dag.apply_operation_back(Measure(), [dag.qubits[qubit]], [dag.clbits[idx]])

        return dag
