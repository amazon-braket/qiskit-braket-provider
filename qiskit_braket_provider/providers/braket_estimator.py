from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import SupportsFloat, SupportsIndex, TypeAlias, cast
from uuid import uuid4

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

from braket.circuits.observables import Sum
from braket.program_sets import CircuitBinding, ParameterSets, ProgramSet
from braket.tasks import ProgramSetQuantumTaskResult, QuantumTask
from braket.tasks.measurement_utils import samples_from_measurements
from braket.tasks.program_set_quantum_task_result import CompositeEntry, MeasuredEntry
from qiskit_braket_provider.providers.adapter import (
    rename_parameter,
    to_braket,
    translate_sparse_pauli_op,
)
from qiskit_braket_provider.providers.braket_backend import BraketBackend
from qiskit_braket_provider.providers.braket_primitive_task import BraketPrimitiveTask
from qiskit_braket_provider.providers.braket_quantum_task import _TASK_STATUS_MAP

_DEFAULT_PRECISION = 0.015625  # Same value as BackendEstimatorV2


@lru_cache(maxsize=4096)
def _translate_pauli_observable(pauli: str) -> object:
    return translate_sparse_pauli_op(SparsePauliOp.from_list([(pauli, 1.0)]))


@dataclass
class BraketEstimatorOptions:
    """Options for :class:`~.BraketEstimator`."""

    default_precision: float = _DEFAULT_PRECISION
    """The default precision to use if no run-level or pub-level precision is specified."""

    abelian_grouping: bool = True
    """
    Whether to group qubit-wise commuting Pauli terms before execution.

    Grouping reduces Braket executions by reconstructing compatible Pauli terms from the same
    measurement shots. This preserves expectation-value semantics but changes finite-shot
    covariance compared with measuring each term independently. The reported standard errors for
    grouped Pauli sums use conservative per-term accumulation and are not fully covariance-aware.
    """


_ESTIMATOR_OPTION_NAMES = frozenset(BraketEstimatorOptions.__dataclass_fields__)


_ParameterIndex: TypeAlias = tuple[int, ...]
_ObservableValue: TypeAlias = SparsePauliOp | dict[str, complex]
_TermRoutesByParam: TypeAlias = dict[_ParameterIndex, dict[str, dict[complex, list[int]]]]
_GroupedPaulis: TypeAlias = list[tuple[str, tuple[str, ...]]]


@dataclass(frozen=True)
class _RoutingTarget:
    pauli: str
    coefficient: complex
    parameter_indices: _ParameterIndex
    positions: tuple[int, ...]
    targets: tuple[int, ...]


@dataclass(frozen=True)
class _MeasurementGroup:
    covering_pauli: str
    routed_terms: tuple[_RoutingTarget, ...]


@dataclass(frozen=True)
class _IdentityContribution:
    coefficient: complex
    positions: tuple[int, ...]


@dataclass(frozen=True)
class _QWCBindingMetadata:
    parameter_index_map: dict[_ParameterIndex, int]
    measurement_groups: tuple[_MeasurementGroup, ...]


# (broadcast_position, observable_index_or_None_for_Sum, parameter_set_index)
_ResultMapEntry: TypeAlias = tuple[int, int | None, int]
_ResultMap: TypeAlias = dict[int, list[_ResultMapEntry]]


@dataclass
class _PubMetadata:
    bindings: list[CircuitBinding]
    num_bindings: int
    binding_to_result_map: _ResultMap
    sum_binding_indices: set[int]
    qwc_binding_metadata: dict[int, _QWCBindingMetadata]
    identity_contributions: tuple[_IdentityContribution, ...]


@dataclass
class _JobMetadata:
    pubs: list[EstimatorPub]
    pub_metadata: list[_PubMetadata]
    pub_precisions: list[float]
    pub_shots: list[int]
    binding_locations: dict[tuple[int, int], "_BindingLocation"]


@dataclass(frozen=True)
class _BindingRecord:
    pub_index: int
    binding_index: int
    binding: CircuitBinding


@dataclass(frozen=True)
class _BindingLocation:
    task_index: int
    result_index: int


class _ConstantResultTask(BasePrimitiveJob[PrimitiveResult[PubResult], JobStatus]):
    """Already-complete primitive job for estimator pubs that need no device execution."""

    def __init__(self, result: PrimitiveResult[PubResult]) -> None:
        super().__init__(job_id=f"constant-result-{uuid4()}")
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


class _CompositePrimitiveTask(BasePrimitiveJob[PrimitiveResult[PubResult], JobStatus]):
    """Primitive job that merges multiple Braket ProgramSet tasks into one result."""

    def __init__(
        self,
        tasks: Sequence[QuantumTask],
        result_translator: Callable[[list[ProgramSetQuantumTaskResult]], PrimitiveResult],
        program_sets: Sequence[ProgramSet],
    ) -> None:
        super().__init__(job_id=";".join(task.id for task in tasks))
        self._tasks = list(tasks)
        self._result_translator = result_translator
        self._program_sets = list(program_sets)
        self._result = None

    @property
    def program_sets(self) -> list[ProgramSet]:
        """The ProgramSets submitted by this primitive job."""
        return self._program_sets

    @property
    def program_set(self) -> ProgramSet:
        """The only ProgramSet submitted by this job."""
        if len(self._program_sets) != 1:
            raise ValueError("Composite primitive job has multiple program sets")
        return self._program_sets[0]

    def result(self) -> PrimitiveResult:
        if self._result is None:
            self._result = self._result_translator([task.result() for task in self._tasks])
        return self._result

    def status(self) -> JobStatus:
        statuses = [self._get_task_status(task) for task in self._tasks]
        if all(status == JobStatus.DONE for status in statuses):
            return JobStatus.DONE
        for terminal_status in (JobStatus.ERROR, JobStatus.CANCELLED):
            if any(status == terminal_status for status in statuses):
                return terminal_status
        for active_status in (JobStatus.RUNNING, JobStatus.VALIDATING, JobStatus.QUEUED):
            if any(status == active_status for status in statuses):
                return active_status
        return JobStatus.INITIALIZING

    def cancel(self) -> None:
        for task in self._tasks:
            task.cancel()

    def job_id(self) -> str:
        return ";".join(task.id for task in self._tasks)

    def done(self) -> bool:
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        return any(self._get_task_status(task) == JobStatus.RUNNING for task in self._tasks)

    def cancelled(self) -> bool:
        return all(self._get_task_status(task) == JobStatus.CANCELLED for task in self._tasks)

    def in_final_state(self) -> bool:
        return all(
            self._get_task_status(task) in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]
            for task in self._tasks
        )

    @staticmethod
    def _get_task_status(task: QuantumTask) -> JobStatus:
        return _TASK_STATUS_MAP[task.state()]


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
        max_program_set_executables: int | None = None,
        options: BraketEstimatorOptions | dict[str, object] | None = None,
        **run_options,
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
            max_program_set_executables (int | None): Optional maximum number of executables to
                include in each submitted ProgramSet. If ``None``, no estimator-level chunking is
                applied. Default: ``None``.
            options (BraketEstimatorOptions | dict[str, object] | None): Estimator options
                controlling default precision and qubit-wise commuting observable grouping.
                Supported keys are ``default_precision`` and ``abelian_grouping``. For
                compatibility with previous provider code, these estimator options can also be
                passed directly as constructor keyword arguments.
                When ``abelian_grouping`` is enabled, compatible Pauli terms can share the same
                measurement shots, reducing executions but changing finite-shot covariance compared
                with independent per-term measurements.
            **run_options: Additional options forwarded to the underlying Braket
                ``device.run(...)`` call.
        """
        if not backend._supports_program_sets:
            raise ValueError("Braket device must support program sets")
        self._backend = backend
        self._verbatim = verbatim
        self._optimization_level = optimization_level
        self._max_program_set_executables = max_program_set_executables
        self._options = BraketEstimator._coerce_estimator_options(options, run_options)
        self._run_options = run_options

    @property
    def options(self) -> BraketEstimatorOptions:
        """Return the estimator options."""
        return self._options

    def run(
        self,
        pubs: Iterable[EstimatorPubLike],
        *,
        precision: float | None = None,
        abelian_grouping: bool | None = None,
    ) -> BasePrimitiveJob[PrimitiveResult[PubResult], JobStatus]:
        """
        Run estimator on the given pubs.

        Args:
            pubs (Iterable[EstimatorPubLike]): An iterable of ``EstimatorPubLike`` objects
                to estimate.
            precision (float | None): Target precision for expectation value estimates.
                If ``None``, the estimator's ``default_precision`` option is used.
            abelian_grouping (bool | None): Whether to group qubit-wise commuting Pauli
                terms before execution. If ``None``, the estimator's ``abelian_grouping``
                option is used. Default: ``None``.

        Returns:
            BasePrimitiveJob: A job object containing the estimator results.

        Notes:
            Qubit-wise commuting Pauli terms can be reconstructed from shared Braket
            measurements when ``abelian_grouping`` is enabled. Use
            ``abelian_grouping=False`` to preserve legacy per-term execution shape for a
            specific run.
        """
        precision = self._options.default_precision if precision is None else precision
        if abelian_grouping is None:
            abelian_grouping = self._options.abelian_grouping
        elif not isinstance(abelian_grouping, bool):
            raise TypeError("abelian_grouping must be a bool")

        coerced_pubs = [EstimatorPub.coerce(pub) for pub in pubs]
        pub_precisions = [
            BraketEstimator._effective_precision(pub, precision) for pub in coerced_pubs
        ]
        pub_shots = [
            BraketEstimator._precision_to_shots(pub_precision) for pub_precision in pub_precisions
        ]

        pub_metadata = []  # Track which bindings belong to which pub
        records_by_shots: dict[int, list[_BindingRecord]] = defaultdict(list)

        for pub_index, (pub, shots) in enumerate(zip(coerced_pubs, pub_shots, strict=True)):
            metadata = self._translate_pub(pub, abelian_grouping=abelian_grouping)
            for binding_index, binding in enumerate(metadata.bindings):
                records_by_shots[shots].append(
                    _BindingRecord(
                        pub_index=pub_index,
                        binding_index=binding_index,
                        binding=binding,
                    )
                )
            pub_metadata.append(metadata)

        binding_locations: dict[tuple[int, int], _BindingLocation] = {}
        job_metadata = _JobMetadata(
            pubs=coerced_pubs,
            pub_metadata=pub_metadata,
            pub_precisions=pub_precisions,
            pub_shots=pub_shots,
            binding_locations=binding_locations,
        )
        if not records_by_shots:
            return _ConstantResultTask(BraketEstimator._translate_result(None, job_metadata))

        tasks: list[QuantumTask] = []
        program_sets: list[ProgramSet] = []
        for shots, records in sorted(records_by_shots.items()):
            for chunk in BraketEstimator._chunk_binding_records(
                records,
                max_executables=self._max_program_set_executables,
            ):
                task_index = len(tasks)
                program_set = ProgramSet(
                    [record.binding for record in chunk],
                    shots_per_executable=shots,
                )
                for result_index, record in enumerate(chunk):
                    binding_locations[record.pub_index, record.binding_index] = _BindingLocation(
                        task_index=task_index,
                        result_index=result_index,
                    )
                program_sets.append(program_set)
                tasks.append(self._backend._device.run(program_set, **self._run_options))

        if len(tasks) == 1:
            return BraketPrimitiveTask(
                tasks[0],
                lambda result: BraketEstimator._translate_result([result], job_metadata),
                program_sets[0],
            )
        return _CompositePrimitiveTask(
            tasks,
            lambda results: BraketEstimator._translate_result(results, job_metadata),
            program_sets,
        )

    @staticmethod
    def _coerce_estimator_options(
        options: BraketEstimatorOptions | dict[str, object] | None,
        run_options: dict[str, object],
    ) -> BraketEstimatorOptions:
        option_values: dict[str, object] = (
            {
                "default_precision": options.default_precision,
                "abelian_grouping": options.abelian_grouping,
            }
            if isinstance(options, BraketEstimatorOptions)
            else dict(options or {})
        )

        for option_name in _ESTIMATOR_OPTION_NAMES:
            if option_name not in run_options:
                continue
            if option_name in option_values:
                raise ValueError(
                    f"Specify estimator option {option_name!r} either in options or as a "
                    "constructor keyword argument, not both"
                )
            option_values[option_name] = run_options.pop(option_name)

        unknown_options = sorted(set(option_values) - _ESTIMATOR_OPTION_NAMES)
        if unknown_options:
            raise ValueError(f"Unsupported estimator options: {unknown_options}")

        default_precision_value = cast(
            "str | SupportsFloat | SupportsIndex",
            option_values.pop("default_precision", _DEFAULT_PRECISION),
        )
        try:
            default_precision = float(default_precision_value)
        except (TypeError, ValueError) as ex:
            raise TypeError("default_precision must be a positive float") from ex
        abelian_grouping = option_values.pop("abelian_grouping", True)
        if not isinstance(abelian_grouping, bool):
            raise TypeError("abelian_grouping must be a bool")
        if default_precision <= 0:
            raise ValueError(f"default_precision must be positive, got: {default_precision}")
        return BraketEstimatorOptions(
            default_precision=default_precision,
            abelian_grouping=abelian_grouping,
        )

    @staticmethod
    def _effective_precision(pub: EstimatorPub, precision: float | None) -> float:
        effective_precision = pub.precision if pub.precision is not None else precision
        if effective_precision is None:
            effective_precision = _DEFAULT_PRECISION
        if effective_precision <= 0:
            raise ValueError(f"Precision must be positive, got: {effective_precision}")
        return effective_precision

    @staticmethod
    def _precision_to_shots(precision: float) -> int:
        return int(np.ceil(1.0 / precision**2))

    @staticmethod
    def _chunk_binding_records(
        records: list[_BindingRecord],
        *,
        max_executables: int | None,
    ) -> Iterable[list[_BindingRecord]]:
        if max_executables is None:
            yield records
            return
        if max_executables <= 0:
            raise ValueError("max_program_set_executables must be positive")

        chunk: list[_BindingRecord] = []
        chunk_executables = 0
        for record in records:
            record_executables = len(record.binding)
            if record_executables > max_executables:
                raise ValueError(
                    "A single CircuitBinding requires "
                    f"{record_executables} executables, which exceeds "
                    f"max_program_set_executables={max_executables}"
                )
            if chunk and chunk_executables + record_executables > max_executables:
                yield chunk
                chunk = []
                chunk_executables = 0
            chunk.append(record)
            chunk_executables += record_executables
        if chunk:
            yield chunk

    def _translate_pub(self, pub: EstimatorPub, *, abelian_grouping: bool) -> _PubMetadata:
        """
        Convert an EstimatorPub to CircuitBindings and result reconstruction metadata.

        Since a CircuitBinding only takes one-dimensional parameter and observable arrays,
        multiple CircuitBindings are necessary to capture all the data in an EstimatorPub,
        whose parameter values and observables can take any broadcastable shapes.

        Each broadcasted (parameter values, observable) pair appears in at most one CircuitBinding.

        Args:
            pub (EstimatorPub): The EstimatorPub to convert.

        Returns:
            _PubMetadata: The circuit bindings and metadata needed to reconstruct the pub result.
        """
        backend = self._backend
        circuit = to_braket(
            pub.circuit,
            qubit_labels=backend.qubit_labels,
            target=backend.target,
            verbatim=self._verbatim,
            optimization_level=self._optimization_level,
        )
        observables_broadcast, param_indices_broadcast = BraketEstimator._broadcast_pub(pub)
        if abelian_grouping:
            return self._grouped_bindings(
                circuit, observables_broadcast, param_indices_broadcast, pub.parameter_values
            )
        return self._per_term_bindings(
            circuit, observables_broadcast, param_indices_broadcast, pub.parameter_values
        )

    @staticmethod
    def _broadcast_pub(pub: EstimatorPub) -> tuple[np.ndarray, np.ndarray]:
        observables = np.asarray(pub.observables)
        param_values = pub.parameter_values
        if not param_values.data:
            return observables, BraketEstimator._empty_parameter_indices(observables.shape)

        parameter_indices = np.empty(param_values.shape, dtype=object)
        for index in np.ndindex(param_values.shape):
            parameter_indices[index] = index
        observables_broadcast, parameter_indices_broadcast = np.broadcast_arrays(
            observables, parameter_indices
        )
        return observables_broadcast, parameter_indices_broadcast

    @staticmethod
    def _per_term_bindings(
        circuit: object,
        observables_broadcast: np.ndarray,
        param_indices_broadcast: np.ndarray,
        param_values: BindingsArray,
    ) -> _PubMetadata:
        observables = observables_broadcast
        obs_keys = {BraketEstimator._make_obs_key(obs): obs for obs in observables.flatten()}

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
            sorted_param_indices = sorted(param_indices)
            param_idx_map = {pk: idx for idx, pk in enumerate(sorted_param_indices)}

            braket_observables = [
                translate_sparse_pauli_op(SparsePauliOp.from_list(obs_keys[ok].items()))
                for ok in matching_obs_keys
            ]
            parameter_sets = (
                BraketEstimator._translate_parameters([
                    param_values[pi] for pi in sorted_param_indices
                ])
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
        return _PubMetadata(
            bindings=bindings,
            num_bindings=len(bindings),
            binding_to_result_map=binding_to_result_map,
            sum_binding_indices=sum_binding_indices,
            qwc_binding_metadata={},
            identity_contributions=(),
        )

    @staticmethod
    def _grouped_bindings(
        circuit: object,
        observables_broadcast: np.ndarray,
        param_indices_broadcast: np.ndarray,
        param_values: BindingsArray,
    ) -> _PubMetadata:
        term_routes_by_param: _TermRoutesByParam = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        identity_routes: dict[complex, list[int]] = defaultdict(list)
        for position, (param_indices_value, obs_value) in enumerate(
            zip(param_indices_broadcast.flatten(), observables_broadcast.flatten(), strict=True)
        ):
            param_indices = cast("_ParameterIndex", param_indices_value)
            obs = cast("_ObservableValue", obs_value)
            for pauli, coefficient in BraketEstimator._observable_terms(obs):
                if BraketEstimator._is_identity(pauli):
                    identity_routes[coefficient].append(position)
                else:
                    term_routes_by_param[param_indices][pauli][coefficient].append(position)

        basis_index_by_param_and_term: dict[_ParameterIndex, dict[str, int]] = {}
        parameter_indices_by_bases: dict[tuple[str, ...], list[_ParameterIndex]] = defaultdict(list)
        group_cache: dict[tuple[str, ...], _GroupedPaulis] = {}
        for param_indices, term_routes in sorted(term_routes_by_param.items(), key=str):
            labels_key = tuple(sorted(term_routes))
            if labels_key not in group_cache:
                group_cache[labels_key] = BraketEstimator._group_paulis(labels_key)
            basis_groups = group_cache[labels_key]
            covering_paulis = tuple(basis for basis, _ in basis_groups)
            term_to_basis = {
                term: basis_index
                for basis_index, (_, terms) in enumerate(basis_groups)
                for term in terms
            }
            basis_index_by_param_and_term[param_indices] = term_to_basis
            parameter_indices_by_bases[covering_paulis].append(param_indices)

        bindings: list[CircuitBinding] = []
        qwc_binding_metadata: dict[int, _QWCBindingMetadata] = {}
        for covering_paulis, unsorted_param_indices in sorted(parameter_indices_by_bases.items()):
            matching_param_indices = sorted(unsorted_param_indices, key=str)
            parameter_sets = (
                BraketEstimator._translate_parameters([
                    param_values[pi] for pi in matching_param_indices
                ])
                if param_values.data
                else None
            )
            binding_idx = len(bindings)
            bindings.append(
                CircuitBinding(
                    circuit,
                    input_sets=parameter_sets,
                    observables=[
                        translate_sparse_pauli_op(SparsePauliOp.from_list([(basis, 1.0)]))
                        for basis in covering_paulis
                    ],
                )
            )

            param_idx_map = {pk: idx for idx, pk in enumerate(matching_param_indices)}
            routed_terms_by_basis: dict[int, list[_RoutingTarget]] = defaultdict(list)
            for param_indices in matching_param_indices:
                for pauli, coefficient_routes in sorted(
                    term_routes_by_param[param_indices].items()
                ):
                    basis_index = basis_index_by_param_and_term[param_indices][pauli]
                    for coefficient, positions in sorted(
                        coefficient_routes.items(),
                        key=lambda item: BraketEstimator._coefficient_sort_key(item[0]),
                    ):
                        routed_terms_by_basis[basis_index].append(
                            _RoutingTarget(
                                pauli=pauli,
                                coefficient=coefficient,
                                parameter_indices=param_indices,
                                positions=tuple(positions),
                                targets=BraketEstimator._pauli_targets(pauli),
                            )
                        )
            qwc_binding_metadata[binding_idx] = _QWCBindingMetadata(
                parameter_index_map=param_idx_map,
                measurement_groups=tuple(
                    _MeasurementGroup(
                        covering_pauli=covering_paulis[basis_index],
                        routed_terms=tuple(targets),
                    )
                    for basis_index, targets in sorted(routed_terms_by_basis.items())
                ),
            )

        identity_contributions = tuple(
            _IdentityContribution(coefficient=coefficient, positions=tuple(positions))
            for coefficient, positions in sorted(
                identity_routes.items(),
                key=lambda item: BraketEstimator._coefficient_sort_key(item[0]),
            )
        )
        return _PubMetadata(
            bindings=bindings,
            num_bindings=len(bindings),
            binding_to_result_map={},
            sum_binding_indices=set(),
            qwc_binding_metadata=qwc_binding_metadata,
            identity_contributions=identity_contributions,
        )

    @staticmethod
    def _make_obs_key(obs_val: _ObservableValue) -> str:
        """Create a hashable key for observable values.

        Args:
            obs_val (SparsePauliOp | dict[str, complex]): A SparsePauliOp observable
                or dict representation

        Returns:
            str: A string representation that can be used as a dictionary key
        """
        return str(sorted(obs_val.items())) if isinstance(obs_val, dict) else str(obs_val)

    @staticmethod
    def _observable_terms(
        obs_val: SparsePauliOp | dict[str, complex],
    ) -> tuple[tuple[str, complex], ...]:
        op = (
            obs_val
            if isinstance(obs_val, SparsePauliOp)
            else SparsePauliOp.from_list(list(obs_val.items()))
        ).simplify()
        return tuple(
            (pauli.to_label(), complex(coefficient))
            for pauli, coefficient in zip(op.paulis, op.coeffs, strict=True)
            if not np.isclose(coefficient, 0.0)
        )

    @staticmethod
    def _is_identity(pauli: str) -> bool:
        return all(pauli_char == "I" for pauli_char in pauli)

    @staticmethod
    def _coefficient_sort_key(value: complex) -> tuple[float, float]:
        return (float(np.real(value)), float(np.imag(value)))

    @staticmethod
    def _real_if_close(value: complex, *, context: str) -> float:
        if not np.isclose(np.imag(value), 0.0):
            raise ValueError(f"{context} produced a complex expectation value: {value}")
        return float(np.real(value))

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
    def _empty_parameter_indices(shape: tuple[int, ...]) -> np.ndarray:
        indices = np.empty(shape, dtype=object)
        indices.fill(())
        return indices

    @staticmethod
    def _group_paulis(paulis: Iterable[str]) -> list[tuple[str, tuple[str, ...]]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        op = SparsePauliOp.from_list([(pauli, 1.0) for pauli in sorted(paulis)])
        for commuting_group in op.group_commuting(qubit_wise=True):
            terms = tuple(pauli.to_label() for pauli in commuting_group.paulis)
            grouped[BraketEstimator._covering_pauli(commuting_group)].extend(terms)
        return [(basis, tuple(sorted(terms))) for basis, terms in sorted(grouped.items())]

    @staticmethod
    def _covering_pauli(group: SparsePauliOp) -> str:
        z_mask = np.logical_or.reduce(group.paulis.z, axis=0)
        x_mask = np.logical_or.reduce(group.paulis.x, axis=0)
        basis = []
        for z_bit, x_bit in zip(reversed(z_mask), reversed(x_mask), strict=True):
            if z_bit and x_bit:
                basis.append("Y")
            elif z_bit:
                basis.append("Z")
            elif x_bit:
                basis.append("X")
            else:
                basis.append("I")
        return "".join(basis)

    @staticmethod
    def _pauli_targets(pauli: str) -> tuple[int, ...]:
        return tuple(qubit for qubit, pauli_char in enumerate(reversed(pauli)) if pauli_char != "I")

    @staticmethod
    def _pauli_statistics(
        program_result: MeasuredEntry,
        pauli: str,
        targets: tuple[int, ...] | None = None,
    ) -> tuple[float, float]:
        targets = BraketEstimator._pauli_targets(pauli) if targets is None else targets
        if not targets:
            return 1.0, 0.0
        observable = _translate_pauli_observable(pauli)
        samples = samples_from_measurements(
            program_result.measurements,
            program_result.measured_qubits,
            observable,
            list(targets),
        )
        expectation = np.mean(samples)
        return float(expectation), BraketEstimator._standard_error(samples, expectation)

    @staticmethod
    def _pauli_expectation(program_result: MeasuredEntry, pauli: str) -> float:
        return BraketEstimator._pauli_statistics(program_result, pauli)[0]

    @staticmethod
    def _measured_entry_standard_error(measured_entry: MeasuredEntry) -> float:
        if measured_entry.observable is None:
            raise ValueError("Measured entry has no observable")
        samples = samples_from_measurements(
            measured_entry.measurements,
            measured_entry.measured_qubits,
            measured_entry.observable,
            measured_entry.observable.targets,
        )
        return BraketEstimator._standard_error(samples)

    @staticmethod
    def _sum_standard_error(program_result: CompositeEntry, param_idx: int) -> float:
        num_observables = len(program_result.observables)
        start = param_idx * num_observables
        return sum(
            BraketEstimator._measured_entry_standard_error(program_result[start + obs_idx])
            for obs_idx in range(num_observables)
        )

    @staticmethod
    def _standard_error(samples: np.ndarray, expectation: float | None = None) -> float:
        if len(samples) == 0:
            return 0.0
        expectation = float(np.mean(samples)) if expectation is None else float(expectation)
        variance = max(0.0, float(np.mean(np.square(samples))) - expectation**2)
        return float(np.sqrt(variance / len(samples)))

    @staticmethod
    def _translate_result(
        task_results: Sequence[ProgramSetQuantumTaskResult] | None, metadata: _JobMetadata
    ) -> PrimitiveResult[PubResult]:
        """
        Reconstruct PrimitiveResult from Braket task results.

        Args:
            task_results (Sequence[ProgramSetQuantumTaskResult] | None): The results of submitted
                Braket program set tasks.
            metadata (_JobMetadata): Metadata needed to reconstruct results, including:
                - circuits: List of QuantumCircuits
                - pub_metadata: List of metadata for each pub
                - pub_precisions: Target precision per PUB
                - pub_shots: Number of shots per PUB

        Returns:
            PrimitiveResult[PubResult]: PrimitiveResult containing PubResult for each pub.
        """

        pub_results = []

        for pub_index, (pub, pub_meta) in enumerate(
            zip(metadata.pubs, metadata.pub_metadata, strict=True)
        ):
            num_bindings = pub_meta.num_bindings
            broadcast_shape = pub.shape
            binding_map = pub_meta.binding_to_result_map
            sum_binding_indices = pub_meta.sum_binding_indices

            evs = np.zeros(broadcast_shape, dtype=float)
            stds = np.zeros(broadcast_shape, dtype=float)
            for identity in pub_meta.identity_contributions:
                contribution = BraketEstimator._real_if_close(
                    identity.coefficient, context="Identity term"
                )
                for position in identity.positions:
                    evs[np.unravel_index(position, broadcast_shape)] += contribution

            for local_binding_idx in range(num_bindings):
                if task_results is None:
                    raise ValueError("Task result is required for non-constant estimator pubs")
                binding_location = metadata.binding_locations.get((pub_index, local_binding_idx))
                if binding_location is None:
                    raise ValueError("No task result location was recorded for estimator binding")
                program_result = cast(
                    "CompositeEntry",
                    task_results[binding_location.task_index][binding_location.result_index],
                )
                num_observables = len(program_result.observables)

                if local_binding_idx in pub_meta.qwc_binding_metadata:
                    qwc_metadata = pub_meta.qwc_binding_metadata[local_binding_idx]
                    expectation_cache = {}
                    for obs_idx, measurement_group in enumerate(qwc_metadata.measurement_groups):
                        for target in measurement_group.routed_terms:
                            param_idx = qwc_metadata.parameter_index_map[target.parameter_indices]
                            cache_key = (param_idx, obs_idx, target.pauli)
                            if cache_key not in expectation_cache:
                                expectation_cache[cache_key] = BraketEstimator._pauli_statistics(
                                    program_result[param_idx * num_observables + obs_idx],
                                    target.pauli,
                                    target.targets,
                                )
                            expectation, standard_error = expectation_cache[cache_key]
                            contribution = BraketEstimator._real_if_close(
                                target.coefficient * expectation,
                                context=f"Pauli term {target.pauli}",
                            )
                            for position in target.positions:
                                array_index = np.unravel_index(position, broadcast_shape)
                                evs[array_index] += contribution
                                stds[array_index] += abs(target.coefficient) * standard_error
                    continue

                for position, obs_idx, param_idx in binding_map[local_binding_idx]:
                    # CircuitBinding returns results organized by parameter sets
                    # For each parameter, we get all observables
                    array_index = np.unravel_index(position, broadcast_shape)
                    if local_binding_idx in sum_binding_indices:
                        evs[array_index] = program_result.expectation(param_idx)
                        stds[array_index] = BraketEstimator._sum_standard_error(
                            program_result,
                            param_idx,
                        )
                    else:
                        if obs_idx is None:
                            raise ValueError("Observable result entries must have an index")
                        measured_entry = program_result[param_idx * num_observables + obs_idx]
                        evs[array_index] = measured_entry.expectation
                        stds[array_index] = BraketEstimator._measured_entry_standard_error(
                            measured_entry
                        )

            pub_results.append(
                PubResult(
                    DataBin(evs=evs, stds=stds, shape=broadcast_shape),
                    metadata={
                        "target_precision": metadata.pub_precisions[pub_index],
                        "shots": metadata.pub_shots[pub_index],
                        "circuit_metadata": pub.circuit.metadata,
                    },
                )
            )

        return PrimitiveResult(pub_results, metadata={"version": 2})
