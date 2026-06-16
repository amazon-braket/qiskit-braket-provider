from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
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
from qiskit.quantum_info import SparsePauliOp

from braket.circuits import Circuit
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

# (broadcast_position, parameter_set_index, coefficient, pauli_support)
_GroupedResultMapEntry: TypeAlias = tuple[int, int, float, tuple[int, ...]]


@dataclass
class _IdentityRouting:
    coeff: float
    positions_and_params: list[tuple[int, object]]


@dataclass
class _PubMetadata:
    num_bindings: int
    binding_to_result_map: _ResultMap
    sum_binding_indices: set[int]
    grouped_binding_map: dict[int, list[_GroupedResultMapEntry]]
    grouped_colmap: dict[int, dict[int, int]]
    identity_routing: list[_IdentityRouting]


@dataclass
class _JobMetadata:
    pubs: list[EstimatorPub]
    pub_metadata: list[_PubMetadata]
    precision: float
    shots: int


class _ConstantResultTask(BasePrimitiveJob[PrimitiveResult[PubResult], JobStatus]):
    """Job returned when every observable in a pub is constant (identity only)."""

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
        abelian_grouping: bool = False,
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
            abelian_grouping (bool): Whether to group mutually qubit-wise-commuting observables so
                that each group is estimated with a single Braket executable per parameter set,
                rather than one executable per observable. When ``False`` every observable is
                estimated with its own executable(s). Qiskit's ``BackendEstimatorV2`` enables the
                equivalent grouping by default. Default: ``False``.
        """
        if not backend._supports_program_sets:
            raise ValueError("Braket device must support program sets")
        self._backend = backend
        self._verbatim = verbatim
        self._optimization_level = optimization_level
        self._abelian_grouping = abelian_grouping
        self._options = options

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float = _DEFAULT_PRECISION
    ) -> BasePrimitiveJob:
        """
        Run estimator on the given pubs.

        Args:
            pubs (Iterable[EstimatorPubLike]): An iterable of ``EstimatorPubLike`` objects
                to estimate.
            precision (float): Target precision for expectation value estimates.
                Default: 0.015625.

        Returns:
            BasePrimitiveJob: A job object containing the estimator results. Normally a
                ``BraketPrimitiveTask``, or a ``_ConstantResultTask`` when every observable is
                constant and no task is submitted.
        """
        coerced_pubs = [EstimatorPub.coerce(pub) for pub in pubs]
        pub_precision = BraketEstimator._pub_precision(coerced_pubs)

        all_bindings = []
        pub_metadata = []

        for pub in coerced_pubs:
            bindings, pub_meta = self._translate_pub(pub)
            all_bindings.extend(bindings)
            pub_metadata.append(pub_meta)

        shots = int(np.ceil(1.0 / (pub_precision if pub_precision is not None else precision) ** 2))
        job_metadata = _JobMetadata(
            pubs=coerced_pubs, pub_metadata=pub_metadata, precision=precision, shots=shots
        )

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

    def _translate_pub(self, pub: EstimatorPub) -> tuple[list[CircuitBinding], _PubMetadata]:
        """
        Convert an EstimatorPub to a list of CircuitBindings and its reconstruction metadata.

        Since a CircuitBinding only takes one-dimensional parameter and observable arrays,
        multiple CircuitBindings are necessary to capture all the data in an EstimatorPub,
        whose parameter values and observables can take any broadcastable shapes.

        Each broadcasted (parameter values, observable) pair appears in at most one CircuitBinding.

        Args:
            pub (EstimatorPub): The EstimatorPub to convert.

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

        obs_groups = defaultdict(list)
        for position, (param_indices, obs) in enumerate(
            zip(param_indices_broadcast.flatten(), observables_broadcast.flatten(), strict=True)
        ):
            obs_groups[BraketEstimator._make_obs_key(obs)].append((position, param_indices))

        bindings: list[CircuitBinding] = []
        binding_to_result_map: _ResultMap = {}
        sum_binding_indices: set[int] = set()
        grouped_binding_map: dict[int, list[_GroupedResultMapEntry]] = {}
        grouped_colmap: dict[int, dict[int, int]] = {}
        identity_routing: list[_IdentityRouting] = []
        processed_obs_keys = set()

        for obs_key in obs_groups:
            if obs_key in processed_obs_keys:
                continue

            param_indices = frozenset(pi for _, pi in obs_groups[obs_key])
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

            if self._abelian_grouping:
                group_identity = self._grouped_bindings(
                    circuit=circuit,
                    matching_obs_keys=matching_obs_keys,
                    obs_keys=obs_keys,
                    obs_groups=obs_groups,
                    param_idx_map=param_idx_map,
                    parameter_sets=parameter_sets,
                    bindings=bindings,
                    grouped_binding_map=grouped_binding_map,
                    grouped_colmap=grouped_colmap,
                )
                identity_routing.extend(group_identity)
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
            grouped_binding_map=grouped_binding_map,
            grouped_colmap=grouped_colmap,
            identity_routing=identity_routing,
        )
        return bindings, pub_metadata

    def _grouped_bindings(
        self,
        *,
        circuit: Circuit,
        matching_obs_keys: list[str],
        obs_keys: dict,
        obs_groups: dict,
        param_idx_map: dict,
        parameter_sets: ParameterSets | None,
        bindings: list[CircuitBinding],
        grouped_binding_map: dict[int, list[_GroupedResultMapEntry]],
        grouped_colmap: dict[int, dict[int, int]],
    ) -> list[_IdentityRouting]:
        """
        Partition Pauli terms into qubit-wise-commuting groups and append one binding per group.

        Identity terms are never measured; their contribution is returned for analytic handling.
        """
        identity_routing: list[_IdentityRouting] = []
        active_labels: set[str] = set()
        label_to_coeff_positions: dict[str, list[tuple[float, list[tuple[int, object]]]]] = (
            defaultdict(list)
        )

        for ok in matching_obs_keys:
            positions = obs_groups[ok]
            for label, coeff in obs_keys[ok].items():
                weight = float(np.real(coeff))
                if not BraketEstimator._pauli_support(label):
                    identity_routing.append(
                        _IdentityRouting(coeff=weight, positions_and_params=positions)
                    )
                else:
                    active_labels.add(label)
                    label_to_coeff_positions[label].append((weight, positions))

        if not active_labels:
            return identity_routing

        groups = SparsePauliOp(sorted(active_labels)).group_commuting(qubit_wise=True)
        label_to_binding: dict[str, int] = {}

        for group in groups:
            basis_label = BraketEstimator._group_basis_label(group)
            representative = translate_sparse_pauli_op(SparsePauliOp(basis_label))
            basis_support = sorted(BraketEstimator._pauli_support(basis_label))
            binding_idx = len(bindings)
            bindings.append(
                CircuitBinding(circuit, input_sets=parameter_sets, observables=[representative])
            )
            grouped_colmap[binding_idx] = {qubit: col for col, qubit in enumerate(basis_support)}
            grouped_binding_map[binding_idx] = []
            for label in group.paulis.to_labels():
                label_to_binding[label] = binding_idx

        for label, coeff_positions in label_to_coeff_positions.items():
            binding_idx = label_to_binding[label]
            support = BraketEstimator._pauli_support(label)
            for weight, positions in coeff_positions:
                grouped_binding_map[binding_idx].extend(
                    (position, param_idx_map[pi], weight, support) for position, pi in positions
                )

        return identity_routing

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
        """Build one binding per term for a set of observables (grouping disabled)."""
        braket_observables = [
            translate_sparse_pauli_op(SparsePauliOp.from_list(obs_keys[ok].items()))
            for ok in matching_obs_keys
        ]

        results: list[tuple[CircuitBinding, list[_ResultMapEntry], bool]] = []
        monomials = []
        for ok, observable in zip(matching_obs_keys, braket_observables, strict=True):
            if isinstance(observable, Sum):
                binding = CircuitBinding(circuit, input_sets=parameter_sets, observables=observable)
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
            obs_idx_map = {ok: idx for idx, (ok, _) in enumerate(monomials)}
            result_entries = [
                (position, obs_idx_map[ok], param_idx_map[pi])
                for ok, _ in monomials
                for position, pi in obs_groups[ok]
            ]
            results.append((binding, result_entries, False))

        return results

    @staticmethod
    def _pauli_support(label: str) -> tuple[int, ...]:
        """Return qubit indices on which a Pauli label acts non-trivially."""
        n = len(label)
        return tuple(q for q in range(n) if label[n - 1 - q] != "I")

    @staticmethod
    def _group_basis_label(group: SparsePauliOp) -> str:
        """Return the shared single-qubit measurement basis of a qubit-wise-commuting group."""
        x = group.paulis.x
        z = group.paulis.z
        num_qubits = x.shape[1]
        chars = []
        for q in range(num_qubits):
            has_x = bool(x[:, q].any())
            has_z = bool(z[:, q].any())
            if has_x and not has_z:
                chars.append("X")
            elif has_x and has_z:
                chars.append("Y")
            elif has_z:
                chars.append("Z")
            else:
                chars.append("I")
        return "".join(reversed(chars))

    @staticmethod
    def _make_obs_key(obs_val: SparsePauliOp | dict[str, float]) -> str:
        """Create a hashable key for observable values."""
        return str(sorted(obs_val.items())) if isinstance(obs_val, dict) else str(obs_val)

    @staticmethod
    def _translate_parameters(param_list: list[BindingsArray]) -> ParameterSets:
        """Translate parameter values to Braket ParameterSets."""
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
        """Reconstruct PrimitiveResult from Braket task results."""
        pub_results = []
        binding_offset = 0

        for pub, pub_meta in zip(metadata.pubs, metadata.pub_metadata, strict=True):
            num_bindings = pub_meta.num_bindings
            broadcast_shape = pub.shape
            binding_map = pub_meta.binding_to_result_map
            sum_binding_indices = pub_meta.sum_binding_indices
            grouped_binding_map = pub_meta.grouped_binding_map
            grouped_colmap = pub_meta.grouped_colmap

            evs = np.zeros(broadcast_shape, dtype=float)

            for term in pub_meta.identity_routing:
                for position, _ in term.positions_and_params:
                    evs[np.unravel_index(position, broadcast_shape)] += term.coeff

            for local_binding_idx in range(num_bindings):
                program_result = task_result[binding_offset + local_binding_idx]

                grouped_result_map_entry = grouped_binding_map.get(local_binding_idx)
                if grouped_result_map_entry is not None:
                    col_map = grouped_colmap[local_binding_idx]
                    for position, param_idx, coeff, support in grouped_result_map_entry:
                        measurements = program_result.entries[param_idx].measurements
                        cols = [col_map[qubit] for qubit in support]
                        parity = measurements[:, cols].sum(axis=1) % 2
                        expectation = float(np.mean(1.0 - 2.0 * parity))
                        evs[np.unravel_index(position, broadcast_shape)] += coeff * expectation
                    continue

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
