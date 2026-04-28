from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit.primitives import (
    BaseSamplerV2,
    BitArray,
    DataBin,
    PrimitiveResult,
    PubResult,
    SamplerPubLike,
)
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from braket.circuits import Circuit
from braket.program_sets import CircuitBinding, ParameterSets, ProgramSet
from braket.tasks import ProgramSetQuantumTaskResult
from qiskit_braket_provider.providers.adapter import rename_parameter, to_braket
from qiskit_braket_provider.providers.braket_backend import BraketBackend
from qiskit_braket_provider.providers.braket_primitive_task import BraketPrimitiveTask

_DEFAULT_SHOTS = 1024  # Same value as BackendSamplerV2


@dataclass
class _JobMetadata:
    pubs: list[SamplerPub]
    parameter_indices: list[np.ndarray | None]
    shots: int


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    start: int


class BraketSampler(BaseSamplerV2):
    """
    Runs provided quantum circuit and observable combinations on Amazon Braket devices
    and returns samples of their outputs.
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
        Initialize the Braket sampler.

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
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = _DEFAULT_SHOTS
    ) -> BraketPrimitiveTask:
        """
        Samples circuits with multiple parameter values.

        Args:
            pubs (Iterable[SamplerPubLike]): An iterable of SamplerPubLike objects to sample.
            shots (int): Number of shots to run for each circuit. Default: 1024.

        Returns:
            BraketPrimitiveTask: A job object containing the sample results.
        """
        coerced_pubs = [SamplerPub.coerce(pub) for pub in pubs]
        pub_shots = BraketSampler._pub_shots(coerced_pubs)
        circuit_bindings = []
        parameter_indices = []
        for pub in coerced_pubs:
            circuit_binding, indices = self._translate_pub(pub)
            circuit_bindings.append(circuit_binding)
            parameter_indices.append(indices)
        shots_per_executable = pub_shots if pub_shots is not None else shots
        program_set = ProgramSet(circuit_bindings, shots_per_executable=shots_per_executable)
        return BraketPrimitiveTask(
            self._backend._device.run(program_set, **self._options),
            lambda result: BraketSampler._translate_result(
                result,
                _JobMetadata(
                    pubs=coerced_pubs,
                    parameter_indices=parameter_indices,
                    shots=shots_per_executable,
                ),
            ),
            program_set,
        )

    @staticmethod
    def _pub_shots(pubs: list[SamplerPub]) -> int:
        shots_values = {pub.shots for pub in pubs}
        if len(shots_values) > 1:
            raise ValueError(f"All pubs must have the same shots, got: {shots_values}")
        return next(iter(shots_values))

    def _translate_pub(self, pub: SamplerPub) -> tuple[CircuitBinding | Circuit, np.ndarray | None]:
        backend = self._backend
        circuit = to_braket(
            pub.circuit,
            qubit_labels=backend.qubit_labels,
            target=backend.target,
            verbatim=self._verbatim,
            optimization_level=self._optimization_level,
        )
        param_values = pub.parameter_values
        if not param_values.data:
            return circuit, None
        param_indices = np.fromiter(np.ndindex(param_values.shape), dtype=object).flatten()
        return CircuitBinding(
            circuit,
            input_sets=BraketSampler._translate_parameters([
                param_values[pi] for pi in param_indices
            ]),
        ), param_indices

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
                - pubs: List of EstimatorPub objects
                - parameter_indices: List of n-dimensional parameter indices
                - shots: Number of shots used

        Returns:
            PrimitiveResult[PubResult]: PrimitiveResult containing PubResult for each pub.
        """
        shots = metadata.shots
        pub_results = []
        for program_result, pub, indices in zip(
            task_result.entries, metadata.pubs, metadata.parameter_indices, strict=True
        ):
            circuit = pub.circuit
            meas_info = [
                _MeasureInfo(
                    creg_name=creg.name,
                    num_bits=(num_bits := creg.size),
                    num_bytes=num_bits // 8 + (num_bits % 8 > 0),
                    start=circuit.find_bit(creg[0]).index if num_bits != 0 else 0,
                )
                for creg in circuit.cregs
            ]
            shape = pub.shape
            measurements = np.array([
                executable_result.measurements for executable_result in program_result
            ])
            arrays = {
                item.creg_name: np.zeros((*shape, shots, item.num_bytes), dtype=np.uint8)
                for item in meas_info
            }
            for i, samples in enumerate(measurements):
                for item in meas_info:
                    start = item.start
                    array = np.flip(
                        np.packbits(
                            samples[:, start : start + item.num_bits], axis=1, bitorder="little"
                        ),
                        axis=-1,
                    )
                    if indices is None:
                        arrays[item.creg_name] = array
                    else:
                        arrays[item.creg_name][indices[i]] = array

            pub_results.append(
                PubResult(
                    DataBin(
                        **{
                            item.creg_name: BitArray(arrays[item.creg_name], item.num_bits)
                            for item in meas_info
                        },
                        shape=shape,
                    ),
                    metadata={"shots": shots, "circuit_metadata": circuit.metadata},
                )
            )
        return PrimitiveResult(pub_results)
