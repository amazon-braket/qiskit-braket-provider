"""Tests for Amazon Braket task."""

from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import pytest
from qiskit.providers import JobStatus

from braket.aws import AwsQuantumTask
from braket.aws.queue_information import QuantumTaskQueueInfo, QueueType
from qiskit_braket_provider import (
    AmazonBraketTask,
    AWSBraketJob,
    BraketAwsBackend,
    BraketLocalBackend,
    BraketQuantumTask,
)
from qiskit_braket_provider.providers.braket_quantum_task import retry_if_result_none
from test.unit_tests.mocks import MOCK_LOCAL_QUANTUM_TASK, MOCK_PROGRAM_SET_QUANTUM_TASK


class TestBraketQuantumTask(TestCase):
    """Tests BraketTask."""

    def test_retry_if_result_none(self) -> None:
        """Test when result is None"""
        assert retry_if_result_none(None) is True

    def _get_task(self) -> BraketQuantumTask:
        return BraketQuantumTask(
            backend=BraketLocalBackend(name="default"),
            task_id="AwesomeId",
            tasks=[MOCK_LOCAL_QUANTUM_TASK],
            shots=10,
        )

    def test_task(self) -> None:
        """Tests task."""
        task = self._get_task()
        self.assertTrue(isinstance(task, BraketQuantumTask))
        self.assertEqual(None, task.submit())  # type: ignore[func-returns-value]
        self.assertEqual(task.shots, 10)

        self.assertEqual(task.status(), JobStatus.DONE)

    def test_result(self) -> None:
        """Tests result."""
        task = self._get_task()

        self.assertEqual(task.result().job_id, "AwesomeId")
        self.assertEqual(task.result().backend_name, "default")
        self.assertEqual(task.result().get_memory(), ["10", "10", "1"])

        self.assertEqual(task.result().results[0].data.counts, {"01": 1, "10": 2})
        self.assertEqual(task.result().results[0].data.memory, ["0x2", "0x2", "0x1"])
        self.assertEqual(task.result().results[0].status, "COMPLETED")
        self.assertEqual(task.result().results[0].shots, 3)

    def test_program_set(self) -> None:
        """Tests program set."""
        task = BraketQuantumTask(
            backend=BraketLocalBackend(name="default"),
            task_id="TaskID",
            tasks=MOCK_PROGRAM_SET_QUANTUM_TASK,
            shots=10,
        )

        self.assertEqual(task.status(), JobStatus.DONE)
        self.assertEqual(task.result().job_id, "TaskID")
        self.assertEqual(task.result().status, "COMPLETED")
        self.assertEqual(
            task.result().results[0].data.counts,
            {"11": 9, "00": 7, "01": 2, "10": 2},
        )
        self.assertEqual(
            task.result().results[0].data.memory,
            [
                "0x0",
                "0x2",
                "0x3",
                "0x0",
                "0x3",
                "0x0",
                "0x3",
                "0x1",
                "0x3",
                "0x0",
                "0x3",
                "0x0",
                "0x2",
                "0x1",
                "0x3",
                "0x3",
                "0x3",
                "0x0",
                "0x3",
                "0x0",
            ],
        )
        self.assertEqual(task.result().results[0].shots, 20)

    @patch(
        "qiskit_braket_provider.providers.braket_quantum_task.AwsQuantumTaskBatch._retrieve_results"
    )
    def test_task_result_is_none(self, mock_retrieve_results: MagicMock) -> None:
        """Tests result when result is None"""
        mock_retrieve_results.return_value = [None, None]

        task = BraketQuantumTask(
            backend=BraketLocalBackend(name="default"),
            task_id="AwesomeId",
            tasks=[MOCK_LOCAL_QUANTUM_TASK, MOCK_LOCAL_QUANTUM_TASK],
            shots=10,
        )
        result = task.result()

        assert result.results == [None, None]

    @patch(
        "qiskit_braket_provider.providers.braket_quantum_task.AwsQuantumTask",
        spec=AwsQuantumTask,
    )
    def test_queue_position(self, mock_aws_quantum_task: MagicMock) -> None:
        """Tests queue position retrival"""
        task = BraketQuantumTask(
            backend=Mock(spec=BraketAwsBackend),
            task_id="arn:aws:braket:::quantum-task/AwesomeId",
            tasks=[mock_aws_quantum_task],
            shots=10,
        )
        mock_aws_quantum_task.return_value.queue_position.return_value = QuantumTaskQueueInfo(
            queue_type=QueueType.NORMAL, queue_position=1, message=None
        )
        task_queue = task.queue_position()

        mock_aws_quantum_task.return_value.queue_position.assert_called_once()
        mock_aws_quantum_task.assert_called_once_with("arn:aws:braket:::quantum-task/AwesomeId")
        assert task_queue

    @patch(
        "qiskit_braket_provider.providers.braket_quantum_task.AwsQuantumTask",
        spec=AwsQuantumTask,
    )
    def test_queue_position_program_set(self, mock_aws_quantum_task: MagicMock) -> None:
        """Tests queue position retrival for program set tasks"""
        task = BraketQuantumTask(
            backend=Mock(spec=BraketAwsBackend),
            task_id="arn:aws:braket:::quantum-task/AwesomeId",
            tasks=mock_aws_quantum_task,
            shots=10,
        )
        mock_aws_quantum_task.queue_position.return_value = QuantumTaskQueueInfo(
            queue_type=QueueType.NORMAL, queue_position=1, message=None
        )
        task_queue = task.queue_position()

        mock_aws_quantum_task.queue_position.assert_called_once()
        assert task_queue

    @patch(
        "qiskit_braket_provider.providers.braket_quantum_task.AwsQuantumTask",
        spec=AwsQuantumTask,
    )
    def test_task_cancellation(self, mock_aws_quantum_task: MagicMock) -> None:
        """Tests task cancellation"""
        task = BraketQuantumTask(
            backend=Mock(spec=BraketAwsBackend),
            task_id="arn:aws:braket:::quantum-task/AwesomeId",
            tasks=[mock_aws_quantum_task],
            shots=10,
        )
        task.cancel()
        mock_aws_quantum_task.cancel.assert_called_once()

    @patch(
        "qiskit_braket_provider.providers.braket_quantum_task.AwsQuantumTask",
        spec=AwsQuantumTask,
    )
    def test_task_cancellation_program_set(self, mock_aws_quantum_task: MagicMock) -> None:
        """Tests task cancellation for program set tasks"""
        task = BraketQuantumTask(
            backend=Mock(spec=BraketAwsBackend),
            task_id="arn:aws:braket:::quantum-task/AwesomeId",
            tasks=mock_aws_quantum_task,
            shots=10,
        )
        task.cancel()
        mock_aws_quantum_task.cancel.assert_called_once()

    def test_queue_position_for_local_quantum_task(self) -> None:
        """Tests job status when multiple task status is present."""
        task = BraketQuantumTask(
            backend=BraketLocalBackend(name="default"),
            task_id="MockId",
            tasks=[MOCK_LOCAL_QUANTUM_TASK],
            shots=100,
        )
        message = "We don't provide queue information for the LocalQuantumTask."
        with pytest.raises(NotImplementedError, match=message):
            task.queue_position()

    def test_queue_position_empty_tasks(self) -> None:
        """Tests queue position returns None for empty tasks list."""
        task = BraketQuantumTask(
            backend=BraketLocalBackend(name="default"),
            task_id="MockId",
            tasks=[],
            shots=100,
        )
        assert task.queue_position() is None


class TestAmazonBraketTask(TestCase):
    """Tests AmazonBraketTask."""

    def _get_task(self) -> AmazonBraketTask:
        return AmazonBraketTask(
            backend=BraketLocalBackend(name="default"),
            task_id="AwesomeId",
            tasks=[MOCK_LOCAL_QUANTUM_TASK],
            shots=10,
        )

    def test_task(self) -> None:
        """Tests task."""
        task = self._get_task()

        self.assertTrue(isinstance(task, AmazonBraketTask))
        self.assertEqual(task.shots, 10)

        self.assertEqual(task.status(), JobStatus.DONE)

    def test_result(self) -> None:
        """Tests result."""
        task = self._get_task()

        self.assertEqual(task.result().job_id, "AwesomeId")
        self.assertEqual(task.result().results[0].data.counts, {"01": 1, "10": 2})
        self.assertEqual(task.result().results[0].data.memory, ["0x2", "0x2", "0x1"])
        self.assertEqual(task.result().results[0].status, "COMPLETED")
        self.assertEqual(task.result().results[0].shots, 3)
        self.assertEqual(task.result().get_memory(), ["10", "10", "1"])


class TestAWSBraketJob(TestCase):
    """Tests AWSBraketJob"""

    def _get_job(self) -> AWSBraketJob:
        return AWSBraketJob(
            backend=BraketLocalBackend(name="default"),
            job_id="AwesomeId",
            tasks=[MOCK_LOCAL_QUANTUM_TASK],
            shots=10,
        )

    def test_aws_job(self) -> None:
        """Tests job."""
        job = self._get_job()

        self.assertTrue(isinstance(job, AWSBraketJob))
        self.assertEqual(job.shots, 10)

        self.assertEqual(job.result().job_id, "AwesomeId")
        self.assertEqual(job.status(), JobStatus.DONE)

    def test_aws_result(self) -> None:
        """Tests result."""
        job = self._get_job()

        self.assertEqual(job.result().job_id, "AwesomeId")
        self.assertEqual(job.result().results[0].data.counts, {"01": 1, "10": 2})
        self.assertEqual(job.result().results[0].data.memory, ["0x2", "0x2", "0x1"])
        self.assertEqual(job.result().results[0].status, "COMPLETED")
        self.assertEqual(job.result().results[0].shots, 3)
        self.assertEqual(job.result().get_memory(), ["10", "10", "1"])


class TestBraketJobStatus:
    """Tests for Amazon Braket job status."""

    def _get_mock_aws_quantum_task(self, status: str) -> AwsQuantumTask:
        """
        Creates a mock AwsQuantumTask with the given status.
        Status can be one of "CREATED", "QUEUED", "RUNNING", "COMPLETED",
        "FAILED", "CANCELLING", "CANCELLED"
        """
        task = Mock(spec=AwsQuantumTask)
        task.state.return_value = status
        return task

    @pytest.mark.parametrize(
        "task_states, expected_status",
        [
            (["COMPLETED", "FAILED"], JobStatus.ERROR),
            (["COMPLETED", "CANCELLED"], JobStatus.CANCELLED),
            (["COMPLETED", "COMPLETED"], JobStatus.DONE),
            (["RUNNING", "RUNNING"], JobStatus.RUNNING),
            (["QUEUED", "QUEUED"], JobStatus.QUEUED),
        ],
    )
    def test_status(self, task_states: list[str], expected_status: JobStatus) -> None:
        """Tests job status when multiple task status is present."""
        job = AWSBraketJob(
            backend=BraketLocalBackend(name="default"),
            job_id="MockId",
            tasks=[MOCK_LOCAL_QUANTUM_TASK],
            shots=100,
        )
        job._tasks = Mock(spec=BraketQuantumTask)
        job._tasks = [self._get_mock_aws_quantum_task(state) for state in task_states]

        assert job.status() == expected_status
