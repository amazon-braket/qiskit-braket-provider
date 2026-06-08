import operator
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import reduce
from typing import TypeAlias

import numpy as np
from qiskit.primitives import BaseEstimatorV2, DataBin, EstimatorPubLike, PrimitiveResult, PubResult
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

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
class _PubMetadata:
    num_bindings: int
    binding_to_result_map: _ResultMap
    sum_binding_indices: set[int]
    qwc_metadata: dict


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
    ) -> BraketPrimitiveTask:
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
            bindings, binding_to_result_map, sum_binding_indices, qwc_metadata = (
                self._translate_pub(pub, abelian_grouping=abelian_grouping)
            )
            all_bindings.extend(bindings)
            pub_metadata.append(
                _PubMetadata(
                    num_bindings=len(bindings),
                    binding_to_result_map=binding_to_result_map,
                    sum_binding_indices=sum_binding_indices,
                    qwc_metadata=qwc_metadata,
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
        self, pub: EstimatorPub, abelian_grouping: bool = True
    ) -> tuple[list[CircuitBinding], _ResultMap, set[int], dict]:
        """
        Convert an EstimatorPub to a list of CircuitBindings.

        Since a CircuitBinding only takes one-dimensional parameter and observable arrays,
        multiple CircuitBindings are necessary to capture all the data in an EstimatorPub,
        whose parameter values and observables can take any broadcastable shapes.

        Each broadcasted (parameter values, observable) pair appears in at most one CircuitBinding.

        Args:
            pub (EstimatorPub): The EstimatorPub to convert.
            abelian_grouping (bool): Toggle qubit-wise commutation grouping optimizations.

        Returns:
            tuple[list[CircuitBinding], _ResultMap, set[int], dict]:
            The circuit bindings, pub shape, a map of binding index to array positions,
            the indices of bindings with Pauli sum observables, and metadata for
            reconstructing grouped (qubit-wise commuting) measurements.
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
        qwc_metadata: dict = {}
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
                all_paulis = []
                for ok in matching_obs_keys:
                    op_from_dict = SparsePauliOp.from_list(list(obs_keys[ok].items()))
                    all_paulis.extend(op_from_dict.paulis)

                # Sort for a stable order so grouping and result-mapping are reproducible
                unique_paulis = sorted(set(all_paulis), key=lambda p: p.to_label())

                identity_paulis = []
                active_paulis = []
                for p in unique_paulis:
                    targets = [i for i, char in enumerate(reversed(p.to_label())) if char != "I"]
                    if not targets:
                        identity_paulis.append(p)
                    else:
                        active_paulis.append(p)

                if active_paulis:
                    active_pauli_list = PauliList(active_paulis)
                    for obs_group in active_pauli_list.group_commuting(qubit_wise=True):
                        binding_idx = len(bindings)

                        cov_z = np.logical_or.reduce(obs_group.z)
                        cov_x = np.logical_or.reduce(obs_group.x)
                        covering_pauli = Pauli((cov_z, cov_x))

                        # One covering observable, in a list as CircuitBinding expects
                        braket_covering_obs = translate_sparse_pauli_op(
                            SparsePauliOp(covering_pauli)
                        )
                        bindings.append(
                            CircuitBinding(
                                circuit,
                                input_sets=parameter_sets,
                                observables=[braket_covering_obs],
                            )
                        )

                        routing_targets = []
                        for ok in matching_obs_keys:
                            op_from_dict = SparsePauliOp.from_list(list(obs_keys[ok].items()))
                            for p, coeff in zip(
                                op_from_dict.paulis, op_from_dict.coeffs, strict=True
                            ):
                                if p in obs_group:
                                    routing_targets.append({
                                        "pauli": p,
                                        "coeff": coeff,
                                        "positions_and_params": obs_groups[ok],
                                    })

                        qwc_metadata[binding_idx] = {
                            "param_idx_map": param_idx_map,
                            "routing_targets": routing_targets,
                        }

                    if identity_paulis:
                        target_binding = len(bindings) - 1
                        id_routing = []
                        for ok in matching_obs_keys:
                            op_from_dict = SparsePauliOp.from_list(list(obs_keys[ok].items()))
                            for p, coeff in zip(
                                op_from_dict.paulis, op_from_dict.coeffs, strict=True
                            ):
                                if p in identity_paulis:
                                    id_routing.append({
                                        "coeff": coeff,
                                        "positions_and_params": obs_groups[ok],
                                    })
                        qwc_metadata[target_binding]["identity_routing"] = id_routing

                elif identity_paulis:
                    # Pure-constant observable: no real measurement needed, but a binding must carry
                    # at least one observable, so attach a throwaway one.
                    binding_idx = len(bindings)
                    n = pub.circuit.num_qubits
                    dummy_obs = translate_sparse_pauli_op(SparsePauliOp("I" * (n - 1) + "Z"))
                    bindings.append(
                        CircuitBinding(circuit, input_sets=parameter_sets, observables=[dummy_obs])
                    )

                    id_routing = []
                    for ok in matching_obs_keys:
                        op_from_dict = SparsePauliOp.from_list(list(obs_keys[ok].items()))
                        for p, coeff in zip(op_from_dict.paulis, op_from_dict.coeffs, strict=True):
                            if p in identity_paulis:
                                id_routing.append({
                                    "coeff": coeff,
                                    "positions_and_params": obs_groups[ok],
                                })

                    qwc_metadata[binding_idx] = {
                        "param_idx_map": param_idx_map,
                        "routing_targets": [],
                        "identity_routing": id_routing,
                    }

            else:
                braket_observables = [
                    translate_sparse_pauli_op(SparsePauliOp.from_list(obs_keys[ok].items()))
                    for ok in matching_obs_keys
                ]
                binding_idx = len(bindings)
                monomials = []
                for ok, observable in zip(matching_obs_keys, braket_observables, strict=True):
                    if isinstance(observable, Sum):
                        bindings.append(
                            CircuitBinding(
                                circuit, input_sets=parameter_sets, observables=observable
                            )
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
        return bindings, binding_to_result_map, sum_binding_indices, qwc_metadata

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
            qwc_metadata = pub_meta.qwc_metadata

            evs = np.zeros(broadcast_shape, dtype=float)
            for local_binding_idx in range(num_bindings):
                program_result = task_result[binding_offset + local_binding_idx]

                if local_binding_idx in qwc_metadata:
                    meta = qwc_metadata[local_binding_idx]
                    param_idx_map = meta["param_idx_map"]
                    measured_qubits = list(range(pub.circuit.num_qubits))

                    for target in meta["routing_targets"]:
                        pauli = target["pauli"]
                        coeff = target["coeff"]

                        for position, param_indices in target["positions_and_params"]:
                            param_set_idx = param_idx_map[param_indices] if param_indices else 0
                            measured_entry = program_result[param_set_idx]
                            measurements = measured_entry.measurements

                            pauli_str = pauli.to_label()
                            targets = [
                                i for i, char in enumerate(reversed(pauli_str)) if char != "I"
                            ]

                            braket_z_obs = reduce(
                                operator.matmul, [Z() for _ in range(len(targets))]
                            )
                            term_expectation = expectation_from_measurements(
                                measurements, measured_qubits, braket_z_obs, targets
                            )

                            flat_idx = np.unravel_index(position, broadcast_shape)
                            evs[flat_idx] += coeff.real * term_expectation

                    if "identity_routing" in meta:
                        for id_target in meta["identity_routing"]:
                            coeff = id_target["coeff"]
                            for position, _ in id_target["positions_and_params"]:
                                flat_idx = np.unravel_index(position, broadcast_shape)
                                evs[flat_idx] += coeff.real * 1.0
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
