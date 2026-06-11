import operator
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from typing import TypeAlias

import numpy as np
from qiskit.primitives import (
    BaseEstimatorV2,
    BasePrimitiveJob,
    DataBin,
    EstimatorPubLike,
    PrimitiveResult,
    PubResult,
)
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.providers import JobStatus
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

from braket.circuits import Circuit
from braket.circuits.observables import Sum, Z
from braket.program_sets import CircuitBinding, ParameterSets, ProgramSet
from braket.tasks import ProgramSetQuantumTaskResult
from braket.tasks.measurement_utils import expectation_from_measurements
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
class _RoutingTarget:
    pauli: Pauli
    coeff: complex
    positions_and_params: list


@dataclass
class _IdentityRouting:
    coeff: complex
    positions_and_params: list


@dataclass
class _QWCBindingMetadata:
    param_idx_map: dict
    routing_targets: list[_RoutingTarget]


@dataclass
class _PubMetadata:
    num_bindings: int
    binding_to_result_map: _ResultMap
    sum_binding_indices: set[int]
    qwc_metadata: dict[int, _QWCBindingMetadata]
    identity_routing: list[_IdentityRouting]


@dataclass
class _JobMetadata:
    pubs: list[EstimatorPub]
    pub_metadata: list[_PubMetadata]
    precision: float
    shots: int


class _ConstantResultTask(BasePrimitiveJob[PrimitiveResult[PubResult], JobStatus]):
    """
    Job returned when every observable is constant (identity only), so there is
    nothing to measure and no Braket task is submitted. The result is computed
    analytically up front, unlike the lazy ``BraketPrimitiveTask``.
    """

    def __init__(self, result: PrimitiveResult[PubResult]) -> None:
        super().__init__(job_id="constant")
        self._result = result

    def result(self) -> PrimitiveResult[PubResult]:
        return self._result

    def status(self) -> JobStatus:
        return JobStatus.DONE

    def cancel(self) -> None:
        pass

    def done(self) -> bool:
        return True

    def running(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def in_final_state(self) -> bool:
        return True


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
        **options,
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
        self,
        pubs: Iterable[EstimatorPubLike],
        *,
        precision: float = _DEFAULT_PRECISION,
        abelian_grouping: bool = True,
    ) -> BasePrimitiveJob:
        """
        Run estimator on the given pubs.

        Args:
            pubs (Iterable[EstimatorPubLike]): An iterable of ``EstimatorPubLike`` objects
                to estimate.
            precision (float): Target precision for expectation value estimates.
                Default: 0.015625.
            abelian_grouping (bool): Whether to group qubit-wise commuting observables so
                each group is measured with a single execution. Default: True.

        Returns:
            BraketPrimitiveTask: A job object containing the estimator results.
        """
        coerced_pubs = [EstimatorPub.coerce(pub) for pub in pubs]
        pub_precision = BraketEstimator._pub_precision(coerced_pubs)

        all_bindings = []
        pub_metadata = []  # Track which bindings belong to which pub

        for pub in coerced_pubs:
            bindings, pub_meta = self._translate_pub(pub, abelian_grouping=abelian_grouping)
            all_bindings.extend(bindings)
            pub_metadata.append(pub_meta)

        shots = int(np.ceil(1.0 / (pub_precision if pub_precision is not None else precision) ** 2))
        job_metadata = _JobMetadata(
            pubs=coerced_pubs, pub_metadata=pub_metadata, precision=precision, shots=shots
        )

        # Every observable is constant: nothing to measure, so skip the device entirely
        # and reconstruct the (purely identity) expectation values analytically.
        if not all_bindings:
            return _ConstantResultTask(BraketEstimator._translate_result(None, job_metadata))

        program_set = ProgramSet(all_bindings, shots_per_executable=shots)
        return BraketPrimitiveTask(
            self._backend._device.run(program_set, **self._options),
            lambda result: BraketEstimator._translate_result(result, job_metadata),
            program_set,
        )

    @staticmethod
    def _pub_precision(pubs: list[EstimatorPub]) -> float:
        precision_values = {pub.precision for pub in pubs}
        if len(precision_values) > 1:
            raise ValueError(f"All pubs must have the same precision, got: {precision_values}")
        return next(iter(precision_values))

    def _translate_pub(
        self, pub: EstimatorPub, abelian_grouping: bool
    ) -> tuple[list[CircuitBinding], _PubMetadata]:
        """
        Convert an EstimatorPub to a list of CircuitBindings and its reconstruction metadata.

        Since a CircuitBinding only takes one-dimensional parameter and observable arrays,
        multiple CircuitBindings are necessary to capture all the data in an EstimatorPub,
        whose parameter values and observables can take any broadcastable shapes.

        Each broadcasted (parameter values, observable) pair appears in at most one CircuitBinding.

        Args:
            pub (EstimatorPub): The EstimatorPub to convert.
            abelian_grouping (bool): Toggle qubit-wise commutation grouping optimizations.

        Returns:
            tuple[list[CircuitBinding], _PubMetadata]: The circuit bindings and the metadata
            needed to reconstruct expectation values for this pub.
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
        sum_binding_indices: set[int] = set()
        qwc_metadata: dict[int, _QWCBindingMetadata] = {}
        identity_routing: list[_IdentityRouting] = []
        processed_obs_keys = set()

        for obs_key in obs_groups:
            if obs_key in processed_obs_keys:
                continue

            param_indices = frozenset(pi for _, pi in obs_groups[obs_key])

            # Find other observables with the same parameter sets to complete the Cartesian product
            matching_obs_keys = [
                ok
                for ok, prs in obs_groups.items()
                if (
                    frozenset(pi for _, pi in prs) == param_indices and ok not in processed_obs_keys
                )
            ]
            processed_obs_keys.update(matching_obs_keys)

            ordered_param_indices = sorted(param_indices, key=str)
            param_idx_map = {pk: idx for idx, pk in enumerate(ordered_param_indices)}

            parameter_sets = (
                BraketEstimator._translate_parameters([
                    param_values[pi] for pi in ordered_param_indices
                ])
                if param_values.data
                else None
            )

            if abelian_grouping:
                grouped_bindings, group_identity = self._grouped_bindings(
                    circuit=circuit,
                    parameter_sets=parameter_sets,
                    matching_obs_keys=matching_obs_keys,
                    obs_keys=obs_keys,
                    obs_groups=obs_groups,
                    param_idx_map=param_idx_map,
                )
                identity_routing.extend(group_identity)
                for binding, meta in grouped_bindings:
                    qwc_metadata[len(bindings)] = meta
                    bindings.append(binding)
            else:
                for binding, result_entries, is_sum in self._per_term_bindings(
                    circuit=circuit,
                    parameter_sets=parameter_sets,
                    matching_obs_keys=matching_obs_keys,
                    obs_keys=obs_keys,
                    obs_groups=obs_groups,
                    param_idx_map=param_idx_map,
                ):
                    binding_idx = len(bindings)
                    bindings.append(binding)
                    binding_to_result_map[binding_idx] = result_entries
                    if is_sum:
                        sum_binding_indices.add(binding_idx)

        pub_metadata = _PubMetadata(
            num_bindings=len(bindings),
            binding_to_result_map=binding_to_result_map,
            sum_binding_indices=sum_binding_indices,
            qwc_metadata=qwc_metadata,
            identity_routing=identity_routing,
        )

        return bindings, pub_metadata

    def _grouped_bindings(
        self,
        *,
        circuit: Circuit,
        parameter_sets: ParameterSets | None,
        matching_obs_keys: list[str],
        obs_keys: dict,
        obs_groups: dict,
        param_idx_map: dict,
    ) -> tuple[list[tuple[CircuitBinding, _QWCBindingMetadata]], list[_IdentityRouting]]:
        """
        Build covering-Pauli bindings for one set of observables sharing parameter sets.

        Qubit-wise commuting terms are measured together through a single covering Pauli.
        Constant (identity) terms are never measured: their expectation value is exactly 1,
        so they are returned as ``_IdentityRouting`` entries for the result builder to
        apply directly.

        Returns:
            tuple[list[tuple[CircuitBinding, _QWCBindingMetadata]], list[_IdentityRouting]]:
            The covering bindings paired with their reconstruction metadata, and the
            identity contributions for this group.
        """
        ops_by_key = {
            ok: SparsePauliOp.from_list(list(obs_keys[ok].items())) for ok in matching_obs_keys
        }
        all_paulis = [p for ok in matching_obs_keys for p in ops_by_key[ok].paulis]

        # Sort for a stable order so grouping and result-mapping are reproducible
        unique_paulis = sorted(set(all_paulis), key=lambda p: p.to_label())

        identity_paulis = []
        active_paulis = []
        for p in unique_paulis:
            if any(char != "I" for char in p.to_label()):
                active_paulis.append(p)
            else:
                identity_paulis.append(p)

        identity_routing: list[_IdentityRouting] = []
        if identity_paulis:
            for ok in matching_obs_keys:
                op_from_dict = ops_by_key[ok]
                for p, coeff in zip(op_from_dict.paulis, op_from_dict.coeffs, strict=True):
                    if p in identity_paulis:
                        identity_routing.append(
                            _IdentityRouting(coeff=coeff, positions_and_params=obs_groups[ok])
                        )

        bindings_with_meta: list[tuple[CircuitBinding, _QWCBindingMetadata]] = []
        if active_paulis:
            active_pauli_list = PauliList(active_paulis)
            for obs_group in active_pauli_list.group_commuting(qubit_wise=True):
                cov_z = np.logical_or.reduce(obs_group.z)
                cov_x = np.logical_or.reduce(obs_group.x)
                covering_pauli = Pauli((cov_z, cov_x))

                # One covering observable, in a list as CircuitBinding expects
                braket_covering_obs = translate_sparse_pauli_op(SparsePauliOp(covering_pauli))
                binding = CircuitBinding(
                    circuit, input_sets=parameter_sets, observables=[braket_covering_obs]
                )

                routing_targets = []
                for ok in matching_obs_keys:
                    op_from_dict = ops_by_key[ok]
                    for p, coeff in zip(op_from_dict.paulis, op_from_dict.coeffs, strict=True):
                        if p in obs_group:
                            routing_targets.append(
                                _RoutingTarget(
                                    pauli=p, coeff=coeff, positions_and_params=obs_groups[ok]
                                )
                            )

                bindings_with_meta.append((
                    binding,
                    _QWCBindingMetadata(
                        param_idx_map=param_idx_map, routing_targets=routing_targets
                    ),
                ))

        return bindings_with_meta, identity_routing

    def _per_term_bindings(
        self,
        *,
        circuit: Circuit,
        parameter_sets: ParameterSets | None,
        matching_obs_keys: list[str],
        obs_keys: dict,
        obs_groups: dict,
        param_idx_map: dict,
    ) -> list[tuple[CircuitBinding, list[_ResultMapEntry], bool]]:
        """
        Build one binding per term for a set of observables (grouping disabled).

        This is the original per-term path. Each ``Sum`` observable gets its own binding,
        and the remaining single-term observables share one binding.

        Returns:
            list[tuple[CircuitBinding, list[_ResultMapEntry], bool]]: ``(binding,
            result_map_entries, is_sum)`` tuples. ``is_sum`` is True when the binding's
            observable is a Braket ``Sum``, so the result builder reads a single summed
            expectation rather than per-monomial values.
        """
        braket_observables = [
            translate_sparse_pauli_op(SparsePauliOp.from_list(obs_keys[ok].items()))
            for ok in matching_obs_keys
        ]

        results: list[tuple[CircuitBinding, list[_ResultMapEntry], bool]] = []
        monomials = []
        for ok, observable in zip(matching_obs_keys, braket_observables, strict=True):
            if isinstance(observable, Sum):
                binding = CircuitBinding(circuit, input_sets=parameter_sets, observables=observable)
                # Map each position in the broadcast to its location in the binding result
                result_entries: list[_ResultMapEntry] = [
                    (position, None, param_idx_map[pi]) for position, pi in obs_groups[ok]
                ]
                results.append((binding, result_entries, True))
            else:
                monomials.append((ok, observable))

        if monomials:
            binding = CircuitBinding(
                circuit,
                input_sets=parameter_sets,
                observables=[obs for _, obs in monomials],
            )
            # Map each position in the broadcast to its location in the binding result
            obs_idx_map = {ok: idx for idx, (ok, _) in enumerate(monomials)}
            result_entries = [
                (position, obs_idx_map[ok], param_idx_map[pi])
                for ok, _ in monomials
                for position, pi in obs_groups[ok]
            ]
            results.append((binding, result_entries, False))

        return results

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
        task_result: ProgramSetQuantumTaskResult | None, metadata: _JobMetadata
    ) -> PrimitiveResult[PubResult]:
        """
        Reconstruct PrimitiveResult from Braket task results.

        Args:
            task_result (ProgramSetQuantumTaskResult | None): The result of a Braket program
                set task, or None when every observable is constant and no task was run.
            metadata (_JobMetadata): Metadata needed to reconstruct results, including:
                - pubs: List of EstimatorPubs
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
            qwc_metadata = pub_meta.qwc_metadata

            evs = np.zeros(broadcast_shape, dtype=float)

            # identity terms have expectation 1, so each adds its coefficient directly (no measurement)
            for term in pub_meta.identity_routing:
                for position, _ in term.positions_and_params:
                    evs[np.unravel_index(position, broadcast_shape)] += term.coeff.real

            for local_binding_idx in range(num_bindings):
                program_result = task_result[binding_offset + local_binding_idx]

                if local_binding_idx in qwc_metadata:
                    meta = qwc_metadata[local_binding_idx]
                    param_idx_map = meta.param_idx_map
                    measured_qubits = list(range(pub.circuit.num_qubits))

                    for target in meta.routing_targets:
                        pauli = target.pauli
                        coeff = target.coeff

                        pauli_str = pauli.to_label()
                        targets = [i for i, char in enumerate(reversed(pauli_str)) if char != "I"]
                        braket_z_obs = reduce(operator.matmul, [Z() for _ in range(len(targets))])

                        for position, param_indices in target.positions_and_params:
                            param_set_idx = param_idx_map[param_indices] if param_indices else 0
                            measured_entry = program_result[param_set_idx]
                            measurements = measured_entry.measurements

                            term_expectation = expectation_from_measurements(
                                measurements, measured_qubits, braket_z_obs, targets
                            )

                            flat_idx = np.unravel_index(position, broadcast_shape)
                            evs[flat_idx] += coeff.real * term_expectation
                else:
                    num_observables = len(program_result.observables)
                    for position, obs_idx, param_idx in binding_map[local_binding_idx]:
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
