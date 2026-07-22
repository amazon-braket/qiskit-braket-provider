"""Tests for BraketPrimitiveTask."""

from collections import Counter
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from qiskit.primitives import PrimitiveResult
from qiskit.providers import JobStatus

from braket.circuits import Circuit
from braket.ir.openqasm import Program
from braket.parametric import FreeParameter
from braket.program_sets import CircuitBinding, ParameterSets, ProgramSet
from braket.task_result import ProgramSetTaskMetadata
from braket.task_result.program_set_executable_result_v1 import (
    ProgramSetExecutableResultMetadata,
)
from braket.task_result.program_set_task_metadata_v1 import ProgramMetadata
from braket.tasks import ProgramSetQuantumTaskResult
from braket.tasks.program_set_quantum_task_result import CompositeEntry, MeasuredEntry
from qiskit_braket_provider.providers.braket_primitive_task import (
    BraketPrimitiveTask,
    run_split_program_set,
)


class TestBraketPrimitiveTask(TestCase):
    """Tests for BraketPrimitiveTask."""

    @staticmethod
    def _entry_executable_count(entry: Circuit | CircuitBinding) -> int:
        return len(entry) if isinstance(entry, CircuitBinding) else 1

    @staticmethod
    def _program_set_result(
        task_id: str,
        program_set: ProgramSet,
    ) -> ProgramSetQuantumTaskResult:
        return ProgramSetQuantumTaskResult(
            entries=[
                CompositeEntry(
                    entries=[
                        MeasuredEntry(
                            measurements=np.array([[0]]),
                            counts=Counter({"0": 1}),
                            probabilities={"0": 1.0},
                            measured_qubits=[0],
                            measurements_from_device=True,
                            probabilities_from_device=False,
                            program="OPENQASM 3.0;",
                            inputs=None,
                            observable=None,
                        )
                        for _ in range(TestBraketPrimitiveTask._entry_executable_count(entry))
                    ],
                    program=Program(source="OPENQASM 3.0;", inputs=None),
                    inputs=ParameterSets(),
                    observables=None,
                    shots_per_executable=program_set.shots_per_executable,
                    additional_metadata=None,
                )
                for entry in program_set.entries
            ],
            task_metadata=ProgramSetTaskMetadata(
                id=task_id,
                deviceId="test-device",
                requestedShots=program_set.total_shots,
                successfulShots=program_set.total_shots,
                programMetadata=[
                    ProgramMetadata(
                        executables=[
                            ProgramSetExecutableResultMetadata()
                            for _ in range(TestBraketPrimitiveTask._entry_executable_count(entry))
                        ]
                    )
                    for entry in program_set.entries
                ],
                deviceParameters=None,
                createdAt="2024-10-15T19:06:58.986Z",
                endedAt="2024-10-15T19:07:00.382Z",
                status="COMPLETED",
                totalFailedExecutables=0,
            ),
            num_executables=program_set.total_executables,
            program_set=program_set,
        )

    def test_status(self):
        """Test task status methods."""
        mock_task = Mock()
        mock_task.id = "test-task-id"
        mock_task.state.return_value = "RUNNING"

        task = BraketPrimitiveTask(mock_task, lambda _result: PrimitiveResult([]), None)

        # Test status methods
        self.assertEqual(task.status(), JobStatus.RUNNING)
        self.assertTrue(task.running())
        self.assertFalse(task.done())
        self.assertFalse(task.cancelled())
        self.assertFalse(task.in_final_state())

        # Test completed state
        mock_task.state.return_value = "COMPLETED"
        self.assertEqual(task.status(), JobStatus.DONE)
        self.assertFalse(task.running())
        self.assertTrue(task.done())
        self.assertFalse(task.cancelled())
        self.assertTrue(task.in_final_state())

        # Test cancelled state
        mock_task.state.return_value = "CANCELLED"
        self.assertEqual(task.status(), JobStatus.CANCELLED)
        self.assertFalse(task.running())
        self.assertFalse(task.done())
        self.assertTrue(task.cancelled())
        self.assertTrue(task.in_final_state())

        # Test cancel method
        task.cancel()
        mock_task.cancel.assert_called_once()

        # Test job_id
        self.assertEqual(task.job_id(), "test-task-id")

    def test_empty_task_list_raises_error(self):
        """Test that an empty task list is rejected."""
        with self.assertRaises(ValueError):
            BraketPrimitiveTask([], lambda _result: PrimitiveResult([]), None)

    def test_aggregate_status(self):
        """Test aggregate status precedence for split primitive tasks."""
        completed_task = Mock()
        completed_task.id = "completed-task"
        completed_task.state.return_value = "COMPLETED"
        failed_task = Mock()
        failed_task.id = "failed-task"
        failed_task.state.return_value = "FAILED"
        task = BraketPrimitiveTask(
            [completed_task, failed_task],
            lambda _result: PrimitiveResult([]),
            None,
        )
        self.assertEqual(task.status(), JobStatus.ERROR)

        queued_task = Mock()
        queued_task.id = "queued-task"
        queued_task.state.return_value = "QUEUED"
        task = BraketPrimitiveTask(
            [completed_task, queued_task],
            lambda _result: PrimitiveResult([]),
            None,
        )
        self.assertEqual(task.status(), JobStatus.INITIALIZING)

    def test_multiple_tasks_are_tracked_and_merged(self):
        """Test that split program set task results are merged before translation."""
        program_set = ProgramSet(
            CircuitBinding(
                Circuit().rx(0, FreeParameter("theta")),
                input_sets={"theta": [0.1, 0.2]},
            ),
            shots_per_executable=10,
        )
        program_sets, index_map = program_set.split(1)
        mock_task_1 = Mock()
        mock_task_1.id = "task-1"
        mock_task_1.state.return_value = "COMPLETED"
        mock_task_1.result.return_value = self._program_set_result("result-1", program_sets[0])
        mock_task_2 = Mock()
        mock_task_2.id = "task-2"
        mock_task_2.state.return_value = "COMPLETED"
        mock_task_2.result.return_value = self._program_set_result("result-2", program_sets[1])
        translated_result = PrimitiveResult([])
        result_translator = Mock(return_value=translated_result)

        task = BraketPrimitiveTask(
            [mock_task_1, mock_task_2],
            result_translator,
            program_set,
            index_map,
        )

        self.assertEqual(task.result(), translated_result)

        self.assertEqual(task.job_id(), "task-1;task-2")
        self.assertEqual(task.tasks, (mock_task_1, mock_task_2))
        result_translator.assert_called_once()
        merged_result = result_translator.call_args.args[0]
        self.assertIsInstance(merged_result, ProgramSetQuantumTaskResult)
        self.assertIs(merged_result.program_set, program_set)
        self.assertEqual(merged_result.num_executables, 2)
        self.assertEqual(len(merged_result), 1)
        self.assertEqual(len(merged_result[0]), 2)
        self.assertEqual(task.status(), JobStatus.DONE)

        task.cancel()
        mock_task_1.cancel.assert_called_once()
        mock_task_2.cancel.assert_called_once()

    def test_run_split_program_set_submits_split_program_sets_with_run_batch(self):
        """Test that split program sets are submitted through device.run_batch."""
        backend = Mock()
        device = Mock()
        backend._device = device
        backend._max_program_set_executables = 1
        program_set = ProgramSet(
            CircuitBinding(
                Circuit().rx(0, FreeParameter("theta")),
                input_sets={"theta": [0.1, 0.2]},
            ),
            shots_per_executable=10,
        )
        mock_task_1 = Mock()
        mock_task_1.id = "task-1"
        mock_task_2 = Mock()
        mock_task_2.id = "task-2"
        batch = Mock()
        batch.tasks = [mock_task_1, mock_task_2]
        device.run_batch.return_value = batch

        tasks, index_map = run_split_program_set(backend, program_set, max_parallel=2)

        device.run_batch.assert_called_once()
        submitted_program_sets = device.run_batch.call_args.args[0]
        self.assertEqual([sub.total_executables for sub in submitted_program_sets], [1, 1])
        self.assertEqual([sub.shots_per_executable for sub in submitted_program_sets], [10, 10])
        self.assertEqual(device.run_batch.call_args.kwargs, {"max_parallel": 2, "shots": -1})
        device.run.assert_not_called()
        self.assertEqual(tasks, [mock_task_1, mock_task_2])
        self.assertEqual(index_map, [[0], [1]])

    def test_run_split_program_set_returns_batch_tasks(self):
        """Test that batch tasks are returned directly."""
        backend = Mock()
        device = Mock()
        backend._device = device
        backend._max_program_set_executables = 2
        program_set = ProgramSet(
            CircuitBinding(
                Circuit().rx(0, FreeParameter("theta")),
                input_sets={"theta": [0.1, 0.2]},
            ),
            shots_per_executable=10,
        )
        mock_task = Mock()
        mock_task.id = "task"
        batch = Mock()
        batch.tasks = [mock_task]
        device.run_batch.return_value = batch

        tasks, index_map = run_split_program_set(backend, program_set)

        device.run_batch.assert_called_once_with([program_set], shots=-1)
        self.assertEqual(tasks, [mock_task])
        self.assertEqual(index_map, [[0, 1]])
