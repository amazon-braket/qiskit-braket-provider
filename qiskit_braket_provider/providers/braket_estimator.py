from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from qiskit.primitives import BaseEstimatorV2, DataBin, EstimatorPubLike, PrimitiveResult, PubResult
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
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

# Reconstruction data for an abelian-grouped binding:
# binding_index -> [(broadcast_position, parameter_set_index, coefficient, pauli_support)]
# where pauli_support is the tuple of qubits the Pauli term acts on non-trivially.
_GroupedResultMapEntry: TypeAlias = tuple[int, int, float, tuple[int, ...]]
_GroupedResultMap: TypeAlias = dict[int, list[_GroupedResultMapEntry]]


@dataclass
class _PubMetadata:
    num_bindings: int
    binding_to_result_map: _ResultMap
    sum_binding_indices: set[int]
    grouped_binding_map: _GroupedResultMap
    grouped_colmap: dict[int, dict[int, int]]


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
            (
                bindings,
                binding_to_result_map,
                sum_binding_indices,
                grouped_binding_map,
                grouped_colmap,
            ) = self._translate_pub(pub)
            all_bindings.extend(bindings)
            pub_metadata.append(
                _PubMetadata(
                    num_bindings=len(bindings),
                    binding_to_result_map=binding_to_result_map,
                    sum_binding_indices=sum_binding_indices,
                    grouped_binding_map=grouped_binding_map,
                    grouped_colmap=grouped_colmap,
                )
            )

        shots = int(np.ceil(1.0 / (pub_precision if pub_precision is not None else precision) ** 2))
        program_set = ProgramSet(all_bindings, shots_per_executable=shots)
        options = {"shots": None, **self._options}
        return BraketPrimitiveTask(
            self._backend._device.run(program_set, **options),
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
    ) -> tuple[
        list[CircuitBinding], _ResultMap, set[int], _GroupedResultMap, dict[int, dict[int, int]]
    ]:
        """
        Convert an EstimatorPub to a list of CircuitBindings.

        Since a CircuitBinding only takes one-dimensional parameter and observable arrays,
        multiple CircuitBindings are necessary to capture all the data in an EstimatorPub,
        whose parameter values and observables can take any broadcastable shapes.

        Each broadcasted (parameter values, observable) pair appears in at most one CircuitBinding.

        Args:
            pub (EstimatorPub): The EstimatorPub to convert.

        Returns:
            tuple[list[CircuitBinding], _ResultMap, set[int], _GroupedResultMap,
            dict[int, dict[int, int]]]:
            The circuit bindings, a map of binding index to array positions for
            ungrouped bindings, the indices of bindings with Pauli sum observables,
            a map of binding index to grouped reconstruction entries (when
            ``abelian_grouping`` is enabled), and per-grouped-binding maps from qubit
            index to measurement column.
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
        grouped_binding_map: _GroupedResultMap = {}
        grouped_colmap: dict[int, dict[int, int]] = {}
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
            parameter_sets = (
                BraketEstimator._translate_parameters([param_values[pi] for pi in param_indices])
                if param_values.data
                else None
            )

            if self._abelian_grouping:
                self._emit_commuting_groups(
                    circuit,
                    matching_obs_keys,
                    obs_keys,
                    obs_groups,
                    param_idx_map,
                    parameter_sets,
                    bindings,
                    grouped_binding_map,
                    grouped_colmap,
                )
                continue

            braket_observables = [
                translate_sparse_pauli_op(SparsePauliOp.from_list(obs_keys[ok].items()))
                for ok in matching_obs_keys
            ]
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
        return (
            bindings,
            binding_to_result_map,
            sum_binding_indices,
            grouped_binding_map,
            grouped_colmap,
        )

    def _emit_commuting_groups(
        self,
        circuit: Circuit,
        matching_obs_keys: list,
        obs_keys: dict,
        obs_groups: dict,
        param_idx_map: dict,
        parameter_sets: ParameterSets | None,
        bindings: list[CircuitBinding],
        grouped_binding_map: _GroupedResultMap,
        grouped_colmap: dict[int, dict[int, int]],
    ) -> None:
        """
        Partition the Pauli terms of the given observables into qubit-wise-commuting groups and
        append one ``CircuitBinding`` (one Braket executable per parameter set) per group.

        Every term in a qubit-wise-commuting group shares the same single-qubit measurement basis,
        so a single basis-representative observable measures all of them at once. The per-term
        expectation values are recovered from the raw measurements during result reconstruction.

        Args:
            circuit: The translated Braket circuit shared by these observables.
            matching_obs_keys (list): Observable keys sharing the same parameter sets.
            obs_keys (dict): Map from observable key to the observable (Pauli label -> coefficient).
            obs_groups (dict): Map from observable key to ``(broadcast_position, param_index)`` pairs.
            param_idx_map (dict): Map from broadcast parameter index to its position in the binding.
            parameter_sets: Translated Braket parameter sets, or ``None`` for non-parameterized pubs.
            bindings (list[CircuitBinding]): The binding list to append to (mutated in place).
            grouped_binding_map (_GroupedResultMap): Reconstruction map to populate (mutated in place).
            grouped_colmap (dict): Per-binding qubit-to-column maps to populate (mutated in place).
        """
        # Collect every unique Pauli label across observables that share these parameter sets.
        unique_labels = list({label for ok in matching_obs_keys for label in obs_keys[ok]})
        groups = SparsePauliOp(unique_labels).group_commuting(qubit_wise=True)

        # One binding per qubit-wise-commuting group, measured in that group's shared basis.
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

        # Route each observable's terms to the binding that measures their group.
        for ok in matching_obs_keys:
            positions = obs_groups[ok]
            for label, coeff in obs_keys[ok].items():
                binding_idx = label_to_binding[label]
                support = BraketEstimator._pauli_support(label)
                weight = float(np.real(coeff))
                grouped_binding_map[binding_idx].extend(
                    (position, param_idx_map[pi], weight, support) for position, pi in positions
                )

    @staticmethod
    def _pauli_support(label: str) -> tuple[int, ...]:
        """Return the qubit indices on which a Pauli ``label`` acts non-trivially.

        Args:
            label (str): A Qiskit Pauli label (little-endian: rightmost character is qubit 0).

        Returns:
            tuple[int, ...]: The qubit indices whose character is not ``"I"``.
        """
        n = len(label)
        return tuple(q for q in range(n) if label[n - 1 - q] != "I")

    @staticmethod
    def _group_basis_label(group: SparsePauliOp) -> str:
        """Return the shared single-qubit measurement basis of a qubit-wise-commuting group.

        Within such a group every non-identity term agrees on each qubit's Pauli, so the basis is
        well defined per qubit (``X``/``Y``/``Z``, or ``I`` where no term acts).

        Args:
            group (SparsePauliOp): A qubit-wise-commuting group of Pauli terms.

        Returns:
            str: A Qiskit Pauli label describing the per-qubit measurement basis.
        """
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
            grouped_binding_map = pub_meta.grouped_binding_map
            grouped_colmap = pub_meta.grouped_colmap

            evs = np.zeros(broadcast_shape, dtype=float)
            for local_binding_idx in range(num_bindings):
                program_result = task_result[binding_offset + local_binding_idx]

                grouped_result_map_entry = grouped_binding_map.get(local_binding_idx)
                if grouped_result_map_entry is not None:
                    # Abelian-grouped binding: recover each term's expectation from raw measurements
                    # via the parity of the bitstring on the term's support, then accumulate the
                    # coefficient-weighted contribution to its observable.
                    col_map = grouped_colmap[local_binding_idx]
                    for position, param_idx, coeff, support in grouped_result_map_entry:
                        if support:
                            measurements = program_result.entries[param_idx].measurements
                            cols = [col_map[qubit] for qubit in support]
                            parity = measurements[:, cols].sum(axis=1) % 2
                            expectation = float(np.mean(1.0 - 2.0 * parity))
                        else:
                            expectation = 1.0  # identity term
                        evs[np.unravel_index(position, broadcast_shape)] += coeff * expectation
                    continue

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
