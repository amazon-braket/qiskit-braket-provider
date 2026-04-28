"""Tests for BraketPrimitiveTask."""

from unittest import TestCase
from unittest.mock import Mock

from qiskit.primitives import PrimitiveResult
from qiskit.providers import JobStatus

from qiskit_braket_provider.providers.braket_primitive_task import BraketPrimitiveTask


class TestBraketPrimitiveTask(TestCase):
    """Tests for BraketPrimitiveTask."""

    def test_status(self) -> None:
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
