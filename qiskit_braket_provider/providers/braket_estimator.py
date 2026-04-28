from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from qiskit.primitives import BaseEstimatorV2, DataBin, EstimatorPubLike, PrimitiveResult, PubResult
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from braket.circuits.observables import Sum
from braket.program_sets import CircuitBinding, ParameterSets, ProgramSet
from braket.tasks import ProgramSetQuantumTaskResult
from qiskit_braket_provider.providers.adapter import (
    rename_parameter,
    to_braket,
    translate_sparse_pauli_op,
)
from qiskit_braket_provider.providers.braket_backend import BraketBackend
from qiskit_braket_provider.providers.braket_primitive_task import BraketPrimitiveTask

_DEFAULT_PRECISION = 0.015625  # Same value as BackendEstimatorV2

# (broadcast_position, observable_index_or_None_for_Sum, parameter_set_index)
_ResultMapEntry: TypeAlias = tuple[int, int | None, int]
_ResultMap: TypeAlias = dict[int, list[_ResultMapEntry]]


@dataclass
class _PubMetadata:
    num_bindings: int
    binding_to_result_map: _ResultMap
    sum_binding_indices: set[int]


@dataclass
class _JobMetadata:
    pubs: list[EstimatorPub]
    pub_metadata: list[_PubMetadata]
    precision: float
    shots: int


class BraketEstimator(BaseEstimatorV2):
    """
    Runs provided quantum circuit and observable combinations on Amazon Braket devices
    and computes their expectation values.
    """

    def __init__(
        self,
        backend: BraketBackend,
        *,
        verbatim: bool = False,
        optimization_level: int = 0,
        **options: Any,  # noqa: ANN401
    ) -> None:
        """
        Initialize the Braket estimator.

        Args:
            backend (BraketBackend): The Braket backend to run circuits on.
            verbatim (bool): Whether to translate the circuit without any modification, in other
                words without transpiling it. Default: ``False``.
            optimization_level (int | None): The optimization level to pass to ``qiskit.transpile``.
                From Qiskit:

                * 0: no optimization - basic translation, no optimization, trivial layout
                * 1: light optimization - routing + potential SaberSwap, some gate cancellation
                  and 1Q gate folding
                * 2: medium optimization - better routing (noise aware) and commutative cancellation
                * 3: high optimization - gate resynthesis and unitary-breaking passes

                Default: 0.
        """
        if not backend._supports_program_sets:
            raise ValueError("Braket device must support program sets")
        self._backend = backend
        self._verbatim = verbatim
        self._optimization_level = optimization_level
        self._options = options

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float = _DEFAULT_PRECISION
    ) -> BraketPrimitiveTask:
        """
        Run estimator on the given pubs.

        Args:
            pubs (Iterable[EstimatorPubLike]): An iterable of ``EstimatorPubLike`` objects
                to estimate.
            precision (float): Target precision for expectation value estimates.
                Default: 0.015625.

        Returns:
            BraketPrimitiveTask: A job object containing the estimator results.
        """
        coerced_pubs = [EstimatorPub.coerce(pub) for pub in pubs]
        pub_precision = BraketEstimator._pub_precision(coerced_pubs)

        all_bindings = []
        pub_metadata = []  # Track which bindings belong to which pub

        for pub in coerced_pubs:
            bindings, binding_to_result_map, sum_binding_indices = self._translate_pub(pub)
            all_bindings.extend(bindings)
            pub_metadata.append(
                _PubMetadata(
                    num_bindings=len(bindings),
                    binding_to_result_map=binding_to_result_map,
                    sum_binding_indices=sum_binding_indices,
                )
            )

        shots = int(np.ceil(1.0 / (pub_precision if pub_precision is not None else precision) ** 2))
        program_set = ProgramSet(all_bindings, shots_per_executable=shots)
        return BraketPrimitiveTask(
            self._backend._device.run(program_set, **self._options),
            lambda result: BraketEstimator._translate_result(
                result,
                _JobMetadata(
                    pubs=coerced_pubs, pub_metadata=pub_metadata, precision=precision, shots=shots
                ),
            ),
            program_set,
        )

    @staticmethod
    def _pub_precision(pubs: list[EstimatorPub]) -> float:
        precision_values = {pub.precision for pub in pubs}
        if len(precision_values) > 1:
            raise ValueError(f"All pubs must have the same precision, got: {precision_values}")
        return next(iter(precision_values))

    def _translate_pub(
        self, pub: EstimatorPub
    ) -> tuple[list[CircuitBinding], _ResultMap, set[int]]:
        """
        Convert an EstimatorPub to a list of CircuitBindings.

        Since a CircuitBinding only takes one-dimensional parameter and observable arrays,
        multiple CircuitBindings are necessary to capture all the data in an EstimatorPub,
        whose parameter values and observables can take any broadcastable shapes.

        Each broadcasted (parameter values, observable) pair appears in at most one CircuitBinding.

        Args:
            pub (EstimatorPub): The EstimatorPub to convert.

        Returns:
            tuple[list[CircuitBinding], _ResultMap, set[int]]:
            The circuit bindings, pub shape, a map of binding index to array positions,
            and the indices of bindings with Pauli sum observables.
        """
        backend = self._backend
        circuit = to_braket(
            pub.circuit,
            qubit_labels=backend.qubit_labels,
            target=backend.target,
            verbatim=self._verbatim,
            optimization_level=self._optimization_level,
        )

        observables = np.asarray(pub.observables)
        param_values = pub.parameter_values
        obs_keys = {BraketEstimator._make_obs_key(obs): obs for obs in observables.flatten()}
        observables_broadcast, param_indices_broadcast = (
            np.broadcast_arrays(
                observables,
                np.fromiter(np.ndindex(shape := param_values.shape), dtype=object).reshape(shape),
            )
            if param_values.data
            else (observables, np.empty(observables.shape, dtype=object))
        )

        # Group parameter sets with the same observable
        obs_groups = defaultdict(list)
        for position, (param_indices, obs) in enumerate(
            zip(param_indices_broadcast.flatten(), observables_broadcast.flatten(), strict=True)
        ):
            obs_groups[BraketEstimator._make_obs_key(obs)].append((position, param_indices))

        bindings: list[CircuitBinding] = []
        binding_to_result_map: _ResultMap = {}
        sum_binding_indices = set()
        processed_obs_keys = set()

        for obs_key, pairs in obs_groups.items():
            if obs_key in processed_obs_keys:
                continue

            param_indices = frozenset(pi for _, pi in pairs)

            # Find other observables with the same parameter sets to complete the Cartesian product
            matching_obs_keys = [
                ok
                for ok, prs in obs_groups.items()
                if (
                    frozenset(pi for _, pi in prs) == param_indices and ok not in processed_obs_keys
                )
            ]
            processed_obs_keys.update(matching_obs_keys)
            param_idx_map = {pk: idx for idx, pk in enumerate(param_indices)}

            braket_observables = [
                translate_sparse_pauli_op(SparsePauliOp.from_list(obs_keys[ok].items()))
                for ok in matching_obs_keys
            ]
            parameter_sets = (
                BraketEstimator._translate_parameters([param_values[pi] for pi in param_indices])
                if param_values.data
                else None
            )
            binding_idx = len(bindings)
            monomials = []
            for ok, observable in zip(matching_obs_keys, braket_observables, strict=True):
                if isinstance(observable, Sum):
                    bindings.append(
                        CircuitBinding(circuit, input_sets=parameter_sets, observables=observable)
                    )
                    # Map each position in the broadcast to its location in the binding result
                    binding_to_result_map[binding_idx] = [
                        (position, None, param_idx_map[pi]) for position, pi in obs_groups[ok]
                    ]
                    sum_binding_indices.add(binding_idx)
                    binding_idx += 1
                else:
                    monomials.append((ok, observable))

            if monomials:
                bindings.append(
                    CircuitBinding(
                        circuit,
                        input_sets=parameter_sets,
                        observables=[obs for _, obs in monomials],
                    )
                )
                # Map each position in the broadcast to its location in the binding result
                obs_idx_map = {ok: idx for idx, (ok, _) in enumerate(monomials)}
                binding_to_result_map[len(bindings) - 1] = [
                    (position, obs_idx_map[ok], param_idx_map[pi])
                    for ok, _ in monomials
                    for position, pi in obs_groups[ok]
                ]
        return bindings, binding_to_result_map, sum_binding_indices

    @staticmethod
    def _make_obs_key(obs_val: SparsePauliOp | dict[str, float]) -> str:
        """Create a hashable key for observable values.

        Args:
            obs_val (SparsePauliOp | dict[str, float]): A SparsePauliOp observable
                or dict representation

        Returns:
            str: A string representation that can be used as a dictionary key
        """
        return str(sorted(obs_val.items())) if isinstance(obs_val, dict) else str(obs_val)

    @staticmethod
    def _translate_parameters(param_list: list[BindingsArray]) -> ParameterSets:
        """
        Translate parameter values to Braket ParameterSets.

        Args:
            param_list (list[BindingsArray]): List of parameter value arrays.

        Returns:
            ParameterSets: Braket ParameterSets object.
        """
        data = defaultdict(list)
        for bindings_array in param_list:
            for k, v in bindings_array.data.items():
                for param, val in zip(k, v, strict=True):
                    data[rename_parameter(param)].append(val)
        return ParameterSets(data)

    @staticmethod
    def _translate_result(
        task_result: ProgramSetQuantumTaskResult, metadata: _JobMetadata
    ) -> PrimitiveResult[PubResult]:
        """
        Reconstruct PrimitiveResult from Braket task results.

        Args:
            task_result (ProgramSetQuantumTaskResult): The result of a Braket program set task
            metadata (_JobMetadata): Metadata needed to reconstruct results, including:
                - circuits: List of QuantumCircuits
                - pub_metadata: List of metadata for each pub
                - precision: Target precision
                - shots: Number of shots used

        Returns:
            PrimitiveResult[PubResult]: PrimitiveResult containing PubResult for each pub.
        """

        pub_results = []
        binding_offset = 0

        for pub, pub_meta in zip(metadata.pubs, metadata.pub_metadata, strict=True):
            num_bindings = pub_meta.num_bindings
            broadcast_shape = pub.shape
            binding_map = pub_meta.binding_to_result_map
            sum_binding_indices = pub_meta.sum_binding_indices

            evs = np.zeros(broadcast_shape, dtype=float)
            for local_binding_idx in range(num_bindings):
                program_result = task_result[binding_offset + local_binding_idx]
                num_observables = len(program_result.observables)

                for position, obs_idx, param_idx in binding_map[local_binding_idx]:
                    # CircuitBinding returns results organized by parameter sets
                    # For each parameter, we get all observables
                    evs[np.unravel_index(position, broadcast_shape)] = (
                        program_result.expectation(param_idx)
                        if local_binding_idx in sum_binding_indices
                        else program_result[param_idx * num_observables + obs_idx].expectation
                    )

            pub_results.append(
                PubResult(
                    DataBin(evs=evs, shape=broadcast_shape),
                    metadata={
                        "target_precision": metadata.precision,
                        "shots": metadata.shots,
                        "circuit_metadata": pub.circuit.metadata,
                    },
                )
            )
            binding_offset += num_bindings

        return PrimitiveResult(pub_results)
