"""Tests for BraketEstimator."""

from collections.abc import Iterable, Sequence
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimatorV2, BasePrimitiveJob
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.providers import JobStatus
from qiskit.quantum_info import SparsePauliOp

from braket.program_sets import ProgramSet
from qiskit_braket_provider.providers import BraketLocalBackend
from qiskit_braket_provider.providers import braket_estimator as braket_estimator_module
from qiskit_braket_provider.providers.braket_estimator import BraketEstimator


class _MockProgramResult:
    """Minimal ProgramSet entry result for estimator result-translation tests."""

    def __init__(self, entries: Sequence[object]) -> None:
        self._entries = list(entries)
        self.observables = [object() for _ in self._entries]

    def __getitem__(self, index: int) -> object:
        return self._entries[index]


class _SizedBinding:
    """Minimal binding object with a configurable executable count."""

    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length


class TestBraketEstimator(TestCase):
    """Tests for BraketEstimator."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = BraketLocalBackend()
        self.estimator = BraketEstimator(self.backend)
        self.ungrouped_estimator = BraketEstimator(
            self.backend, options={"abelian_grouping": False}
        )
        self.estimator_backend = BackendEstimatorV2(backend=self.backend)

    def assert_correct_results(self, task: BasePrimitiveJob, pubs: Iterable[EstimatorPubLike]):
        """Compares the results from BraketEstimator and BackendEstimatorV2"""
        for actual, expected in zip(
            task.result(), self.estimator_backend.run(pubs).result(), strict=True
        ):
            self.assertTrue(np.allclose(actual.data.evs, expected.data.evs, rtol=0.3, atol=0.2))

    @staticmethod
    def _mock_measured_entry(bits: object) -> Mock:
        entry = Mock()
        entry.measurements = np.asarray(bits, dtype=int)
        entry.measured_qubits = [0]
        return entry

    @staticmethod
    def _mock_task(task_id: str, task_result: object, state: str = "COMPLETED") -> Mock:
        task = Mock()
        task.id = task_id
        task.result.return_value = task_result
        task.state.return_value = state
        return task

    @staticmethod
    def _non_empty_circuit(num_qubits: int) -> QuantumCircuit:
        circuit = QuantumCircuit(num_qubits)
        circuit.h(0)
        return circuit

    def test_program_sets_unsupported(self):
        """Tests that initialization raises a ValueError if program sets aren't supported"""
        backend = BraketLocalBackend()
        backend._supports_program_sets = False
        with self.assertRaises(ValueError):
            BraketEstimator(backend)

    def test_simple_pub(self):
        """Test a simple pub with no broadcasting."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        observable = SparsePauliOp(["ZZ"])
        pub = (qc, observable)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            self.assertIsInstance(call_args[0][0], ProgramSet)
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_parameterized_circuit(self):
        """Test with a parameterized circuit."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        observable = SparsePauliOp(["Z"])
        param_values = np.array([[0.0], [np.pi / 4], [np.pi / 2]])
        pub = (qc, observable, param_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_multiple_observables(self):
        """Test with multiple observables."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        observables = [SparsePauliOp(["ZZ"]), SparsePauliOp(["XX"])]
        pub = (qc, observables)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            program_set = mock_run.call_args[0][0]
            self.assertIsInstance(program_set, ProgramSet)

    def test_multiple_pubs(self):
        """Test running multiple pubs."""
        qc1 = QuantumCircuit(1)
        qc1.h(0)

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)

        obs1 = SparsePauliOp(["Z"])
        obs2 = SparsePauliOp(["ZZ"])

        pub1 = (qc1, obs1)
        pub2 = (qc2, obs2)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub1, pub2], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_default_precision(self):
        """Test that default precision is used when not specified."""
        qc = QuantumCircuit(1)
        qc.h(0)
        observable = SparsePauliOp(["Z"])
        pub = (qc, observable)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub])

            self.assertIsInstance(job, BasePrimitiveJob)

    def test_constructor_default_precision_option(self):
        """Tests Qiskit-style default_precision estimator option routing."""
        estimator = BraketEstimator(self.backend, options={"default_precision": 0.05})
        qc = QuantumCircuit(1)
        qc.h(0)
        pub = (qc, SparsePauliOp(["Z"]))

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = estimator.run([pub])

        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args[0][0].shots_per_executable, 400)
        self.assertIsInstance(job, BasePrimitiveJob)

    def test_options_property_and_validation(self):
        """Tests estimator option access and validation errors."""
        options = braket_estimator_module.BraketEstimatorOptions(
            default_precision=0.05,
            abelian_grouping=False,
        )
        estimator = BraketEstimator(self.backend, options=options)

        self.assertEqual(estimator.options, options)

        with self.assertRaises(ValueError) as duplicate_context:
            BraketEstimator(
                self.backend,
                options={"default_precision": 0.05},
                default_precision=0.1,
            )
        self.assertIn("either in options", str(duplicate_context.exception))

        with self.assertRaises(ValueError) as unknown_context:
            BraketEstimator(self.backend, options={"unsupported": True})
        self.assertIn("Unsupported estimator options", str(unknown_context.exception))

        with self.assertRaises(TypeError):
            BraketEstimator(self.backend, options={"default_precision": object()})

        with self.assertRaises(TypeError):
            BraketEstimator(self.backend, options={"abelian_grouping": "false"})

        with self.assertRaises(ValueError):
            BraketEstimator(self.backend, options={"default_precision": 0.0})

    def test_custom_precision(self):
        """Test using custom precision."""
        qc = QuantumCircuit(1)
        qc.h(0)
        observable = SparsePauliOp(["Z"])
        pub = (qc, observable)

        custom_precision = 0.05

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=custom_precision)

            self.assertIsInstance(job, BasePrimitiveJob)

    def test_constructor_abelian_grouping_option_uses_legacy_path(self):
        """Tests Qiskit-style abelian_grouping estimator option routing."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        pub = (circuit, [SparsePauliOp("ZZ"), SparsePauliOp("ZI"), SparsePauliOp("IZ")])

        task = self.ungrouped_estimator.run([pub])

        self.assertEqual(len(task.program_set), 1)
        self.assertEqual(len(task.program_set[0]), 3)
        self.assert_correct_results(task, [pub])

    def test_direct_abelian_grouping_constructor_keyword_not_forwarded(self):
        """Tests legacy constructor keyword routing for abelian_grouping."""
        estimator = BraketEstimator(self.backend, abelian_grouping=False)
        qc = self._non_empty_circuit(1)
        pub = (qc, SparsePauliOp(["Z"]))

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            estimator.run([pub])

        mock_run.assert_called_once()
        self.assertNotIn("abelian_grouping", mock_run.call_args.kwargs)

    def test_run_abelian_grouping_keyword_uses_legacy_path(self):
        """Tests upstream-style run-level abelian_grouping routing."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        pub = (circuit, [SparsePauliOp("ZZ"), SparsePauliOp("ZI"), SparsePauliOp("IZ")])

        task = self.estimator.run([pub], abelian_grouping=False)

        self.assertEqual(len(task.program_set), 1)
        self.assertEqual(len(task.program_set[0]), 3)
        self.assert_correct_results(task, [pub])

    def test_run_abelian_grouping_keyword_overrides_constructor_option(self):
        """Tests that a run-level grouping option overrides the estimator default."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        pub = (circuit, [SparsePauliOp("ZZ"), SparsePauliOp("ZI"), SparsePauliOp("IZ")])

        task = self.ungrouped_estimator.run([pub], abelian_grouping=True)

        self.assertEqual(len(task.program_set), 1)
        self.assertEqual(len(task.program_set[0]), 1)
        self.assert_correct_results(task, [pub])

    def test_run_rejects_invalid_abelian_grouping_keyword(self):
        """Tests explicit validation for the run-level abelian_grouping option."""
        qc = QuantumCircuit(1)
        pub = (qc, SparsePauliOp(["Z"]))

        with self.assertRaises(TypeError):
            self.estimator.run([pub], abelian_grouping="false")

    def test_device_run_options_still_forwarded(self):
        """Tests non-estimator constructor kwargs still route to device.run."""
        estimator = BraketEstimator(self.backend, disable_qubit_rewiring=True)
        qc = self._non_empty_circuit(1)
        pub = (qc, SparsePauliOp(["Z"]))

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            estimator.run([pub])

        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.kwargs, {"disable_qubit_rewiring": True})

    def test_complex_broadcasting(self):
        """Test with complex broadcasting shapes (2, 3, 6)."""
        theta = Parameter("θ")
        phi = Parameter("φ")
        qc = QuantumCircuit(2)
        qc.ry(theta, 0)
        qc.rx(phi, 1)
        qc.cx(0, 1)

        # Parameter values with shape (3, 6)
        rng = np.random.default_rng(42)
        param_data = {
            "θ": rng.uniform(0, 2 * np.pi, size=(3, 6)),
            "φ": rng.uniform(0, 2 * np.pi, size=(3, 6)),
        }
        parameter_values = BindingsArray(param_data)

        # Observables with shape (2, 3, 1)
        observables = [
            [[SparsePauliOp(["ZZ"])], [SparsePauliOp(["XX"])], [SparsePauliOp(["YY"])]],
            [[SparsePauliOp(["ZI"])], [SparsePauliOp(["IZ"])], [SparsePauliOp(["XI"])]],
        ]

        pub = (qc, observables, parameter_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            program_set = mock_run.call_args[0][0]
            self.assertIsInstance(program_set, ProgramSet)

            self.assertIsInstance(job, BasePrimitiveJob)

    def test_broadcasting_with_scalar_observable(self):
        """Test broadcasting with scalar observable and array parameters."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        param_values = np.linspace(0, np.pi, 5)
        observable = SparsePauliOp(["Z"])
        pub = (qc, observable, param_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_broadcasting_with_array_observables(self):
        """Test broadcasting with array observables and scalar parameters."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        observables = [
            SparsePauliOp(["ZZ"]),
            SparsePauliOp(["XX"]),
            SparsePauliOp(["YY"]),
            SparsePauliOp(["ZI"]),
        ]

        pub = (qc, observables)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_different_precisions_use_shot_buckets(self):
        """Test that pubs with different precisions are submitted in shot buckets."""
        qc = QuantumCircuit(1)
        qc.h(0)
        observable = SparsePauliOp(["Z"])

        obs_array = ObservablesArray([observable])
        pub1 = EstimatorPub(circuit=qc, observables=obs_array, precision=0.01)
        pub2 = EstimatorPub(circuit=qc, observables=obs_array, precision=0.02)

        task_1 = Mock()
        task_1.id = "test-task-id-1"
        task_2 = Mock()
        task_2.id = "test-task-id-2"
        with patch.object(self.backend._device, "run", side_effect=[task_1, task_2]) as mock_run:
            job = self.estimator.run([pub1, pub2])

        self.assertIsInstance(job, BasePrimitiveJob)
        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(
            [call_args[0][0].shots_per_executable for call_args in mock_run.call_args_list],
            [2500, 10000],
        )
        self.assertEqual(
            [program_set.shots_per_executable for program_set in job.program_sets],
            [2500, 10000],
        )
        with self.assertRaises(ValueError):
            _ = job.program_set

    def test_different_precisions_result_aggregates_shot_buckets(self):
        """Tests composite result aggregation across mixed-precision shot buckets."""
        circuit = self._non_empty_circuit(1)
        observable = SparsePauliOp("Z")
        observables = ObservablesArray([observable])
        pub_10000_shots = EstimatorPub(circuit=circuit, observables=observables, precision=0.01)
        pub_2500_shots = EstimatorPub(circuit=circuit, observables=observables, precision=0.02)

        task_2500_shots = self._mock_task(
            "test-task-id-2500",
            [_MockProgramResult([self._mock_measured_entry([[0], [0], [0], [0]])])],
        )
        task_10000_shots = self._mock_task(
            "test-task-id-10000",
            [_MockProgramResult([self._mock_measured_entry([[1], [1], [1], [1]])])],
        )
        with patch.object(
            self.backend._device,
            "run",
            side_effect=[task_2500_shots, task_10000_shots],
        ) as mock_run:
            job = self.estimator.run([pub_10000_shots, pub_2500_shots])

        self.assertEqual(
            [call_args[0][0].shots_per_executable for call_args in mock_run.call_args_list],
            [2500, 10000],
        )

        results = job.result()
        self.assertEqual(results.metadata, {"version": 2})
        self.assertTrue(np.allclose(results[0].data.evs, [-1.0]))
        self.assertTrue(np.allclose(results[1].data.evs, [1.0]))
        self.assertTrue(np.allclose(results[0].data.stds, [0.0]))
        self.assertTrue(np.allclose(results[1].data.stds, [0.0]))
        self.assertEqual(results[0].metadata["target_precision"], 0.01)
        self.assertEqual(results[0].metadata["shots"], 10000)
        self.assertEqual(results[1].metadata["target_precision"], 0.02)
        self.assertEqual(results[1].metadata["shots"], 2500)

        self.assertIs(job.result(), results)
        self.assertEqual(task_2500_shots.result.call_count, 1)
        self.assertEqual(task_10000_shots.result.call_count, 1)

    def test_constant_pub_metadata_uses_effective_precision(self):
        """Tests per-PUB precision metadata for analytical constant results."""
        circuit = self._non_empty_circuit(1)
        observables = ObservablesArray([SparsePauliOp("I")])
        pub1 = EstimatorPub(circuit=circuit, observables=observables, precision=0.01)
        pub2 = EstimatorPub(circuit=circuit, observables=observables, precision=None)

        with patch.object(self.backend._device, "run") as mock_run:
            job = self.estimator.run([pub1, pub2], precision=0.02)

        mock_run.assert_not_called()
        results = job.result()
        self.assertEqual(results.metadata, {"version": 2})
        self.assertEqual(results[0].metadata["target_precision"], 0.01)
        self.assertEqual(results[0].metadata["shots"], 10000)
        self.assertEqual(results[1].metadata["target_precision"], 0.02)
        self.assertEqual(results[1].metadata["shots"], 2500)
        self.assertTrue(np.allclose(results[0].data.stds, 0.0))
        self.assertTrue(np.allclose(results[1].data.stds, 0.0))

    def test_max_program_set_executables_chunks_submissions(self):
        """Tests optional ProgramSet chunking by executable count."""
        estimator = BraketEstimator(self.backend, max_program_set_executables=1)
        qc = self._non_empty_circuit(1)
        pubs = [(qc, SparsePauliOp("Z")), (qc, SparsePauliOp("X"))]

        task_1 = Mock()
        task_1.id = "test-task-id-1"
        task_2 = Mock()
        task_2.id = "test-task-id-2"
        with patch.object(self.backend._device, "run", side_effect=[task_1, task_2]) as mock_run:
            job = estimator.run(pubs, precision=0.1)

        self.assertIsInstance(job, BasePrimitiveJob)
        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(
            [call_args[0][0].total_executables for call_args in mock_run.call_args_list],
            [1, 1],
        )
        self.assertEqual(len(job.program_sets), 2)

    def test_chunked_program_sets_result_aggregates(self):
        """Tests composite result aggregation across chunked ProgramSet submissions."""
        estimator = BraketEstimator(self.backend, max_program_set_executables=1)
        circuit = self._non_empty_circuit(1)
        pubs = [(circuit, SparsePauliOp("Z")), (circuit, SparsePauliOp("X"))]

        z_task = self._mock_task(
            "test-task-id-z",
            [_MockProgramResult([self._mock_measured_entry([[0], [0], [0], [0]])])],
        )
        x_task = self._mock_task(
            "test-task-id-x",
            [_MockProgramResult([self._mock_measured_entry([[1], [1], [1], [1]])])],
        )
        with patch.object(self.backend._device, "run", side_effect=[z_task, x_task]):
            job = estimator.run(pubs, precision=0.1)

        results = job.result()
        self.assertEqual(results.metadata, {"version": 2})
        self.assertTrue(np.allclose(results[0].data.evs, 1.0))
        self.assertTrue(np.allclose(results[1].data.evs, -1.0))
        self.assertTrue(np.allclose(results[0].data.stds, 0.0))
        self.assertTrue(np.allclose(results[1].data.stds, 0.0))

    def test_chunking_rejects_non_positive_limits(self):
        """Tests explicit validation for invalid ProgramSet chunk limits."""
        for max_executables in (0, -1):
            with self.subTest(max_executables=max_executables):
                with self.assertRaises(ValueError) as context:
                    list(
                        BraketEstimator._chunk_binding_records(
                            [],
                            max_executables=max_executables,
                        )
                    )
                self.assertIn("must be positive", str(context.exception))

    def test_chunking_rejects_binding_larger_than_limit(self):
        """Tests validation when one CircuitBinding exceeds the chunk executable limit."""
        record = Mock()
        record.binding = _SizedBinding(2)

        with self.assertRaises(ValueError) as context:
            list(BraketEstimator._chunk_binding_records([record], max_executables=1))

        self.assertIn("exceeds max_program_set_executables=1", str(context.exception))

    def test_effective_precision_defaults_and_validation(self):
        """Tests direct effective precision fallback and validation."""
        pub = EstimatorPub.coerce((QuantumCircuit(1), SparsePauliOp("Z")))

        self.assertEqual(BraketEstimator._effective_precision(pub, None), 0.015625)
        with self.assertRaises(ValueError):
            BraketEstimator._effective_precision(pub, 0.0)

    def test_composite_task_result_status_and_controls(self):
        """Tests composite primitive task result caching and status helpers."""
        program_set = object()
        task = self._mock_task("done-task", "raw-result")
        result_translator = Mock(return_value="translated-result")
        job = braket_estimator_module._CompositePrimitiveTask(
            [task],
            result_translator,
            [program_set],
        )

        self.assertIs(job.program_set, program_set)
        self.assertEqual(job.job_id(), "done-task")
        self.assertEqual(job.result(), "translated-result")
        self.assertEqual(job.result(), "translated-result")
        task.result.assert_called_once()
        result_translator.assert_called_once_with(["raw-result"])
        self.assertEqual(job.status(), JobStatus.DONE)
        self.assertTrue(job.done())
        self.assertFalse(job.running())
        self.assertFalse(job.cancelled())
        self.assertTrue(job.in_final_state())

        job.cancel()
        task.cancel.assert_called_once()

    def test_composite_task_status_precedence(self):
        """Tests composite primitive task status precedence across Braket states."""
        completed_task = self._mock_task("completed-task", object(), state="COMPLETED")
        failed_task = self._mock_task("failed-task", object(), state="FAILED")
        running_task = self._mock_task("running-task", object(), state="RUNNING")
        cancelled_task = self._mock_task("cancelled-task", object(), state="CANCELLED")
        initialized_task = self._mock_task("initialized-task", object(), state="INITIALIZED")

        failed_job = braket_estimator_module._CompositePrimitiveTask(
            [completed_task, failed_task],
            Mock(),
            [object(), object()],
        )
        self.assertEqual(failed_job.status(), JobStatus.ERROR)
        self.assertTrue(failed_job.in_final_state())

        running_job = braket_estimator_module._CompositePrimitiveTask(
            [completed_task, running_task],
            Mock(),
            [object(), object()],
        )
        self.assertEqual(running_job.status(), JobStatus.RUNNING)
        self.assertTrue(running_job.running())
        self.assertFalse(running_job.in_final_state())

        cancelled_job = braket_estimator_module._CompositePrimitiveTask(
            [cancelled_task],
            Mock(),
            [object()],
        )
        self.assertEqual(cancelled_job.status(), JobStatus.CANCELLED)
        self.assertTrue(cancelled_job.cancelled())

        initialized_job = braket_estimator_module._CompositePrimitiveTask(
            [initialized_task],
            Mock(),
            [object()],
        )
        self.assertEqual(initialized_job.status(), JobStatus.INITIALIZING)

    def test_non_broadcastable_shapes_raises_error(self):
        """Test that non-broadcastable shapes raise an error."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        # Create observables with shape (3,)
        observables = [SparsePauliOp(["Z"]), SparsePauliOp(["X"]), SparsePauliOp(["Y"])]

        # Create parameter values with shape (2, 1) - not broadcastable with (3,)
        param_values = np.array([[0.0], [np.pi / 4]])

        pub = (qc, observables, param_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            with self.assertRaises(ValueError) as context:
                self.estimator.run([pub], precision=0.01)

            self.assertIn("not broadcastable", str(context.exception))

    def test_run_local_single_observable_or_parameter(self):
        """Tests that correct results are returned when there is only one observable or parameter"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("θ"), 0)
        observables = [SparsePauliOp("ZX"), SparsePauliOp("XZ")]
        parameters = [np.pi / 3, np.pi / 6]
        pubs = [
            (circuit, observables[0], parameters),
            (circuit, observables, parameters[0]),
            (circuit, observables[1], parameters[1]),
        ]
        task = self.ungrouped_estimator.run(pubs)
        program_set = task.program_set
        self.assertEqual(len(program_set), 3)
        self.assertEqual(len(program_set[0]), 2)
        self.assertEqual(len(program_set[1]), 2)
        self.assertEqual(len(program_set[2]), 1)
        self.assert_correct_results(task, pubs)

    def test_run_local_no_parameters(self):
        """Tests that correct results are returned for circuits with no parameters"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(np.pi / 3, 0)
        observables = [SparsePauliOp("ZX"), SparsePauliOp("XZ")]
        pubs = [(circuit, observables), (circuit, observables[0])]
        task = self.ungrouped_estimator.run(pubs)
        program_set = task.program_set
        self.assertEqual(len(program_set), 2)
        self.assertEqual(len(program_set[0]), 2)
        self.assertEqual(len(program_set[1]), 1)
        self.assert_correct_results(task, pubs)

    def test_run_local_pauli_sum(self):
        """Tests that correct results are returned when one observable is a Pauli sum"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("θ[0]"), 0)
        circuit.rz(Parameter("θ[1]"), 0)
        circuit.cx(0, 1)
        circuit.h(0)

        num_params = 20
        pub = (
            circuit,
            [
                [SparsePauliOp("ZZ")],
                [SparsePauliOp("ZX")],
                [SparsePauliOp("XZ")],
                [SparsePauliOp(["ZX", "XZ"], [0.3, 0.8])],
            ],
            np.vstack([
                np.linspace(-np.pi, np.pi, num_params),
                np.linspace(-4 * np.pi, 4 * np.pi, num_params),
            ]).T,
        )

        task = self.ungrouped_estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 2)
        self.assertEqual(len(program_set[0]), num_params * 2)
        self.assertEqual(len(program_set[1]), num_params * 3)
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping(self):
        """Tests that commuting Pauli terms are grouped into representative executions."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        pub = (
            circuit,
            [
                SparsePauliOp("ZZ"),
                SparsePauliOp("ZI"),
                SparsePauliOp("IZ"),
                SparsePauliOp(["ZZ", "ZI"], [0.5, -0.25]),
                SparsePauliOp("XX"),
            ],
        )

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 2)
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping_broadcasting(self):
        """Tests grouped results with broadcasted parameters and Pauli sums."""
        theta = Parameter("theta")
        circuit = QuantumCircuit(2)
        circuit.ry(theta, 0)
        circuit.cx(0, 1)

        num_params = 8
        pub = (
            circuit,
            [
                [SparsePauliOp("ZZ")],
                [SparsePauliOp(["ZI", "IZ"], [0.3, 0.7])],
                [SparsePauliOp("XX")],
            ],
            np.linspace(0, np.pi, num_params),
        )

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), num_params * 2)
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping_group_collapse(self):
        """Tests that Z-basis QWC terms collapse into one covering executable."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        pub = (circuit, [SparsePauliOp("ZZ"), SparsePauliOp("ZI"), SparsePauliOp("IZ")])

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 1)
        result = task.result()[0]
        self.assertEqual(result.data.stds.shape, result.data.evs.shape)
        self.assertTrue(np.all(result.data.stds >= 0.0))
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping_shared_terms(self):
        """Tests that shared Pauli terms are reconstructed from one covering measurement."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        pub = (
            circuit,
            [
                SparsePauliOp(["ZI", "IZ"], [0.5, 0.25]),
                SparsePauliOp(["ZI", "ZZ"], [-0.75, 0.5]),
            ],
        )

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 1)
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping_identity_with_active_terms(self):
        """Tests that identity terms are added analytically without extra executions."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        pub = (circuit, SparsePauliOp(["II", "ZZ"], [0.5, 1.25]))

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 1)
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping_pure_identity(self):
        """Tests that pure identity observables complete without submitting a Braket task."""
        circuit = self._non_empty_circuit(2)
        observable = SparsePauliOp("II") * 2.5
        pub = (circuit, observable)

        with patch.object(self.backend._device, "run") as mock_run:
            job = self.estimator.run([pub])

        mock_run.assert_not_called()
        self.assertEqual(job.status(), JobStatus.DONE)
        self.assertTrue(job.done())
        self.assertTrue(job.in_final_state())
        self.assertFalse(job.running())
        self.assertFalse(job.cancelled())
        self.assertIsNone(job.cancel())
        result = job.result()[0]
        self.assertTrue(np.allclose(result.data.evs, 2.5))
        self.assertTrue(np.allclose(result.data.stds, 0.0))

    def test_run_local_abelian_grouping_constant_pub_with_active_pub(self):
        """Tests that mixed constant and active pubs submit only active grouped work."""
        constant_circuit = self._non_empty_circuit(2)
        active_circuit = QuantumCircuit(2)
        active_circuit.h(0)
        active_circuit.cx(0, 1)

        constant_pub = (constant_circuit, SparsePauliOp("II") * 3.0)
        active_pub = (active_circuit, [SparsePauliOp("ZZ"), SparsePauliOp("ZI")])

        task = self.estimator.run([constant_pub, active_pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 1)

        results = task.result()
        self.assertTrue(np.allclose(results[0].data.evs, 3.0))
        self.assert_correct_results(task, [constant_pub, active_pub])

    def test_run_local_abelian_grouping_broadcast_identity_only(self):
        """Tests that broadcasted identity-only observables stay fully analytical."""
        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)

        pub = (
            circuit,
            [[SparsePauliOp("I") * 1.5], [SparsePauliOp("I") * -2.0]],
            np.array([0.0, np.pi / 4, np.pi / 2]),
        )

        with patch.object(self.backend._device, "run") as mock_run:
            job = self.estimator.run([pub])

        mock_run.assert_not_called()
        expected = np.array([[1.5, 1.5, 1.5], [-2.0, -2.0, -2.0]])
        result = job.result()[0]
        self.assertEqual(result.data.evs.shape, (2, 3))
        self.assertEqual(result.data.stds.shape, (2, 3))
        self.assertTrue(np.allclose(result.data.evs, expected))
        self.assertTrue(np.allclose(result.data.stds, 0.0))

    def test_grouped_bindings_zero_observable_after_simplify(self):
        """Tests that provider grouping treats zero-simplified observables as constant."""
        observable = SparsePauliOp(["Z", "Z"], [1.0, -1.0])
        observables = np.empty((), dtype=object)
        observables[()] = observable
        parameter_indices = BraketEstimator._empty_parameter_indices(())

        metadata = BraketEstimator._grouped_bindings(
            object(), observables, parameter_indices, BindingsArray()
        )

        self.assertEqual(metadata.num_bindings, 0)
        self.assertEqual(metadata.bindings, [])
        self.assertEqual(metadata.identity_contributions, ())

    def test_run_local_abelian_grouping_rejects_zero_observable_after_qiskit_simplify(self):
        """Tests that Qiskit container validation rejects zero-simplified observables."""
        circuit = QuantumCircuit(1)
        circuit.h(0)
        observable = SparsePauliOp(["Z", "Z"], [1.0, -1.0])
        pub = (circuit, observable)

        with (
            patch.object(self.backend._device, "run") as mock_run,
            self.assertRaises(ValueError) as context,
        ):
            self.estimator.run([pub])

        mock_run.assert_not_called()
        self.assertIn("Empty observable was detected", str(context.exception))

    def test_grouped_bindings_reuses_grouping_plan_for_matching_terms(self):
        """Tests grouped parameter slices reuse the same QWC grouping plan."""
        observables = np.empty((2,), dtype=object)
        observables[:] = [SparsePauliOp(["ZI", "IZ"], [0.5, 0.25])] * 2
        parameter_indices = np.empty((2,), dtype=object)
        parameter_indices[:] = [(0,), (1,)]

        with (
            patch.object(
                BraketEstimator,
                "_group_paulis",
                wraps=BraketEstimator._group_paulis,
            ) as group_paulis,
            patch.object(braket_estimator_module, "CircuitBinding", return_value=Mock()),
        ):
            metadata = BraketEstimator._grouped_bindings(
                object(), observables, parameter_indices, BindingsArray()
            )

        self.assertEqual(metadata.num_bindings, 1)
        self.assertEqual(group_paulis.call_count, 1)

    def test_pauli_statistics_computes_expectation_from_samples(self):
        """Tests Pauli statistics only processes measurement samples once."""
        measured_entry = Mock()
        measured_entry.measurements = np.array([[0], [1], [0], [0]])
        measured_entry.measured_qubits = [0]
        observable = Mock()
        samples = np.array([1.0, -1.0, 1.0, 1.0])

        with (
            patch.object(
                braket_estimator_module,
                "_translate_pauli_observable",
                return_value=observable,
            ) as translate_observable,
            patch.object(
                braket_estimator_module,
                "samples_from_measurements",
                return_value=samples,
            ) as samples_from_measurements,
        ):
            expectation, standard_error = BraketEstimator._pauli_statistics(
                measured_entry, "Z", (0,)
            )

        translate_observable.assert_called_once_with("Z")
        samples_from_measurements.assert_called_once_with(
            measured_entry.measurements,
            measured_entry.measured_qubits,
            observable,
            [0],
        )
        self.assertEqual(expectation, 0.5)
        self.assertAlmostEqual(
            standard_error,
            np.sqrt(1 - expectation**2) / np.sqrt(len(samples)),
        )
        self.assertAlmostEqual(
            BraketEstimator._standard_error(2.0 * samples),
            2.0 * np.sqrt(1 - expectation**2) / np.sqrt(len(samples)),
        )
        self.assertEqual(BraketEstimator._pauli_statistics(measured_entry, "I"), (1.0, 0.0))
        self.assertEqual(BraketEstimator._standard_error(np.array([])), 0.0)

    def test_measured_entry_standard_error_rejects_missing_observable(self):
        """Tests measured entry standard-error validation."""
        measured_entry = Mock()
        measured_entry.observable = None

        with self.assertRaises(ValueError) as context:
            BraketEstimator._measured_entry_standard_error(measured_entry)

        self.assertIn("no observable", str(context.exception))

    def test_translate_result_rejects_missing_task_results(self):
        """Tests result translation validation when task results are absent."""
        pub = EstimatorPub.coerce((QuantumCircuit(1), SparsePauliOp("Z")))
        metadata = braket_estimator_module._JobMetadata(
            pubs=[pub],
            pub_metadata=[
                braket_estimator_module._PubMetadata(
                    bindings=[],
                    num_bindings=1,
                    binding_to_result_map={0: [(0, 0, 0)]},
                    sum_binding_indices=set(),
                    qwc_binding_metadata={},
                    identity_contributions=(),
                )
            ],
            pub_precisions=[0.1],
            pub_shots=[100],
            binding_locations={},
        )

        with self.assertRaises(ValueError) as context:
            BraketEstimator._translate_result(None, metadata)

        self.assertIn("Task result is required", str(context.exception))

    def test_translate_result_rejects_missing_binding_location(self):
        """Tests result translation validation when binding locations are absent."""
        pub = EstimatorPub.coerce((QuantumCircuit(1), SparsePauliOp("Z")))
        metadata = braket_estimator_module._JobMetadata(
            pubs=[pub],
            pub_metadata=[
                braket_estimator_module._PubMetadata(
                    bindings=[],
                    num_bindings=1,
                    binding_to_result_map={0: [(0, 0, 0)]},
                    sum_binding_indices=set(),
                    qwc_binding_metadata={},
                    identity_contributions=(),
                )
            ],
            pub_precisions=[0.1],
            pub_shots=[100],
            binding_locations={},
        )

        with self.assertRaises(ValueError) as context:
            BraketEstimator._translate_result([[]], metadata)

        self.assertIn("No task result location", str(context.exception))

    def test_translate_result_rejects_missing_observable_index(self):
        """Tests result translation validation for malformed observable result maps."""
        pub = EstimatorPub.coerce((QuantumCircuit(1), SparsePauliOp("Z")))
        program_result = _MockProgramResult([Mock()])
        metadata = braket_estimator_module._JobMetadata(
            pubs=[pub],
            pub_metadata=[
                braket_estimator_module._PubMetadata(
                    bindings=[],
                    num_bindings=1,
                    binding_to_result_map={0: [(0, None, 0)]},
                    sum_binding_indices=set(),
                    qwc_binding_metadata={},
                    identity_contributions=(),
                )
            ],
            pub_precisions=[0.1],
            pub_shots=[100],
            binding_locations={(0, 0): braket_estimator_module._BindingLocation(0, 0)},
        )

        with self.assertRaises(ValueError) as context:
            BraketEstimator._translate_result([[program_result]], metadata)

        self.assertIn("Observable result entries", str(context.exception))

    def test_run_local_abelian_grouping_identity_with_parameter_sweep(self):
        """Tests analytical identity routing alongside active terms across parameters."""
        theta = Parameter("theta")
        circuit = QuantumCircuit(1)
        circuit.ry(theta, 0)

        num_params = 6
        pub = (
            circuit,
            SparsePauliOp(["I", "Z"], [1.25, -0.5]),
            np.linspace(0, np.pi, num_params),
        )

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), num_params)
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping_dict_observable(self):
        """Tests grouped reconstruction for dictionary observables."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        pub = (circuit, {"ZZ": 0.5, "ZI": -0.25, "IZ": 0.75})

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 1)
        self.assert_correct_results(task, [pub])

    def test_real_if_close_accepts_negligible_imaginary_part(self):
        """Tests that near-real contributions are accepted explicitly."""
        value = BraketEstimator._real_if_close(1.25 + 1e-12j, context="test")

        self.assertEqual(value, 1.25)

    def test_real_if_close_rejects_complex_contribution(self):
        """Tests that complex grouped or identity contributions are rejected explicitly."""
        with self.assertRaises(ValueError) as context:
            BraketEstimator._real_if_close(1.0 + 0.5j, context="test")

        self.assertIn("complex expectation value", str(context.exception))

    def test_run_local_abelian_grouping_y_basis(self):
        """Tests grouped reconstruction for Y-basis covering measurements."""
        circuit = QuantumCircuit(2)
        circuit.rx(np.pi / 4, 0)
        circuit.ry(np.pi / 3, 1)

        pub = (circuit, [SparsePauliOp("YY"), SparsePauliOp("YI"), SparsePauliOp("IY")])

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 1)
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping_mixed_basis(self):
        """Tests grouped reconstruction for compatible mixed-basis Pauli terms."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.ry(np.pi / 5, 1)

        pub = (circuit, [SparsePauliOp("ZI"), SparsePauliOp("IX"), SparsePauliOp("ZX")])

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 1)
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping_multiple_qwc_groups(self):
        """Tests that non-QWC terms split into multiple covering measurements."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        pub = (circuit, SparsePauliOp(["ZI", "XI"], [0.5, 0.5]))

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 2)
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping_splits_global_commuting_non_qwc_terms(self):
        """Tests that globally commuting XX and YY terms remain separate QWC executions."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        pub = (circuit, SparsePauliOp(["XX", "YY"], [0.5, 0.5]))

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(program_set.total_executables, 2)
        self.assertEqual(len(program_set[0]), 2)
        self.assert_correct_results(task, [pub])

    def test_run_local_abelian_grouping_four_qubit_reduces_executions(self):
        """Tests that grouped execution uses fewer executables on a four-qubit workload."""
        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(2, 3)

        pub = (
            circuit,
            [
                SparsePauliOp(["ZZII", "ZIII", "IZII"], [0.4, 0.2, -0.1]),
                SparsePauliOp(["IIXX", "IIXI", "IIIX"], [0.3, -0.2, 0.5]),
                SparsePauliOp(["ZZXX", "ZIXX"], [0.1, 0.7]),
            ],
        )

        grouped_task = self.estimator.run([pub])
        ungrouped_task = self.ungrouped_estimator.run([pub])
        self.assertLess(
            grouped_task.program_set.total_executables,
            ungrouped_task.program_set.total_executables,
        )
        self.assert_correct_results(grouped_task, [pub])

    def test_run_local_abelian_grouping_2d_broadcasting(self):
        """Tests grouped reconstruction across a two-dimensional broadcast result."""
        theta = Parameter("theta")
        circuit = QuantumCircuit(2)
        circuit.ry(theta, 0)
        circuit.cx(0, 1)

        pub = (
            circuit,
            [[SparsePauliOp("ZZ")], [SparsePauliOp("XX")]],
            np.array([[0.0, np.pi / 4, np.pi / 2]]),
        )

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 6)
        self.assertEqual(task.result()[0].data.evs.shape, (2, 3))
        self.assert_correct_results(task, [pub])

    def test_pauli_expectation_uses_measured_qubits_metadata(self):
        """Tests that grouped reconstruction respects Braket measurement column metadata."""
        measured_entry = Mock()
        measured_entry.measurements = np.array([[0, 0], [0, 0], [0, 1], [0, 1]])
        measured_entry.measured_qubits = [1, 0]

        self.assertAlmostEqual(BraketEstimator._pauli_expectation(measured_entry, "ZI"), 1.0)

    def test_run_local_all_pauli_sums(self):
        """Tests that correct results are returned when all observables are Pauli sums"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("theta"), 0)
        circuit.rz(Parameter("phi"), 0)
        circuit.cx(0, 1)
        circuit.h(0)

        num_params = 20
        pub = (
            circuit,
            [
                [SparsePauliOp(["XX", "IY"], [0.5, 0.5])],
                [SparsePauliOp(["YY", "ZI", "XY"], [0.5, 0.5, 0.1])],
            ],
            np.vstack([
                np.linspace(-np.pi, np.pi, num_params),
                np.linspace(-4 * np.pi, 4 * np.pi, num_params),
            ]).T,
        )

        task = self.ungrouped_estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 2)
        self.assertEqual(len(program_set[0]), num_params * 2)
        self.assertEqual(len(program_set[1]), num_params * 3)
        self.assert_correct_results(task, [pub])

    def test_run_local_broadcasting(self):
        """Tests that correct results are returned with broadcasted arrays"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("θ"), 0)
        circuit.h(2)

        num_steps = 6
        pub = (
            circuit,
            [
                [[SparsePauliOp(["IZZ"])], [SparsePauliOp(["IZX"])], [SparsePauliOp(["III"])]],
                [[SparsePauliOp(["IXZ"])], [SparsePauliOp(["IXX"])], [SparsePauliOp(["YYY"])]],
            ],
            np.array(  # shape (3, 6)
                [
                    np.linspace(0, 2 * np.pi, num_steps),
                    np.linspace(0, np.pi, num_steps),
                    np.linspace(np.pi, 2 * np.pi, num_steps),
                ]
            ),
        )

        task = self.ungrouped_estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 3)
        for entry in task.program_set:
            self.assertEqual(len(entry), num_steps * 2)
        self.assert_correct_results(task, [pub])
