"""Transpiler pass to add basis rotation gates for Braket result type pragmas."""

import math

import numpy as np
from qiskit.circuit import Clbit, QuantumCircuit
from qiskit.circuit.library import HGate, RYGate, SGate, UnitaryGate, ZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from braket.ir.jaqcd import (
    Amplitude,
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)

_BASIS_INVARIANT_TYPES = (StateVector, DensityMatrix, Amplitude)
_Z_BASIS_TYPES = (Probability,)
_OBSERVABLE_TYPES = (Expectation, Sample, Variance)


def _rotation_gates_for_observable(observable: str, target: int) -> list[tuple]:
    """Return (gate, qubit) pairs for rotating from the given observable basis to Z basis."""
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
            return []


def _rotation_gates_for_hermitian(matrix: list, targets: list[int]) -> list[tuple]:
    """Compute rotation gates for a Hermitian observable from its eigenvectors."""
    np_matrix = np.array([[complex(c[0], c[1]) for c in row] for row in matrix])
    eigenvalues, eigenvectors = np.linalg.eigh(np_matrix)
    unitary = eigenvectors.conj().T
    return [(UnitaryGate(unitary), targets)]


class AddBasisRotationGates(TransformationPass):
    """Append basis rotation gates and measurements for result type pragmas.

    Reads ``braket_result_pragmas`` from the circuit metadata and appends
    the appropriate rotation gates to change from the observable's eigenbasis
    to the computational (Z) basis, followed by measurements on all targeted qubits.

    For basis-invariant result types (state_vector, density_matrix, amplitude),
    no gates or measurements are added. For Z-basis types (probability), only
    measurements are added. For observable types (expectation, sample, variance),
    rotation gates are added before measurements.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Add basis rotation gates and measurements to the DAG."""
        circuit = dag_to_circuit(dag)
        metadata = circuit.metadata or {}
        result_pragmas = metadata.get("braket_result_pragmas", [])

        if not result_pragmas:
            return dag

        qubits_to_measure = set()
        rotation_ops = []

        for pragma_entry in result_pragmas:
            parsed = pragma_entry["parsed"]

            if isinstance(parsed, _BASIS_INVARIANT_TYPES):
                continue

            if isinstance(parsed, _Z_BASIS_TYPES):
                targets = parsed.targets
                if targets is not None:
                    qubits_to_measure.update(targets)
                else:
                    qubits_to_measure.update(range(circuit.num_qubits))
                continue

            if isinstance(parsed, _OBSERVABLE_TYPES):
                observable = parsed.observable
                targets = parsed.targets

                if targets is None:
                    targets = list(range(circuit.num_qubits))

                obs_idx = 0
                for obs in observable:
                    if isinstance(obs, str):
                        if obs_idx < len(targets):
                            target = targets[obs_idx]
                            rotation_ops.extend(
                                _rotation_gates_for_observable(obs, target)
                            )
                            qubits_to_measure.add(target)
                            obs_idx += 1
                    elif isinstance(obs, list):
                        num_qubits_for_obs = int(math.log2(len(obs)))
                        obs_targets = targets[obs_idx : obs_idx + num_qubits_for_obs]
                        rotation_ops.extend(
                            _rotation_gates_for_hermitian(obs, obs_targets)
                        )
                        qubits_to_measure.update(obs_targets)
                        obs_idx += num_qubits_for_obs

        if not qubits_to_measure and not rotation_ops:
            return dag

        num_clbits_needed = len(qubits_to_measure)
        if circuit.num_clbits < num_clbits_needed:
            circuit.add_bits(
                [Clbit() for _ in range(num_clbits_needed - circuit.num_clbits)]
            )

        for gate, target in rotation_ops:
            if isinstance(target, list):
                circuit.append(gate, target)
            else:
                circuit.append(gate, [target])

        for idx, qubit in enumerate(sorted(qubits_to_measure)):
            circuit.measure(qubit, idx)

        return circuit_to_dag(circuit)
