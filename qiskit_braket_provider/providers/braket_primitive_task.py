from collections.abc import Callable

from qiskit.primitives import BasePrimitiveJob, PrimitiveResult, PubResult
from qiskit.providers import JobStatus

from braket.devices.local_simulator import LocalSimulator
from braket.program_sets import ProgramSet
from braket.tasks import ProgramSetQuantumTaskResult, QuantumTask
from braket.tasks.local_quantum_task import LocalQuantumTask
from qiskit_braket_provider.providers.braket_backend import BraketBackend
from qiskit_braket_provider.providers.braket_quantum_task import _TASK_STATUS_MAP

_TASK_ID_DIVIDER = ";"


class BraketPrimitiveTask(BasePrimitiveJob[PrimitiveResult[PubResult], JobStatus]):
    """
    Job class for Braket-native primitives.

    This class wraps a Braket QuantumTask and constructs a PrimitiveResult
    from the ProgramSetQuantumTaskResult.
    """

    def __init__(
        self,
        task: QuantumTask | list[QuantumTask],
        result_translator: Callable[[ProgramSetQuantumTaskResult], PrimitiveResult],
        program_set: ProgramSet,
        index_map: list[list[int]] | None = None,
    ) -> None:
        """
        Initialize the task.

        Args:
            task (QuantumTask | list[QuantumTask]): The Braket QuantumTask(s)
            result_translator (Callable[[ProgramSetQuantumTaskResult], PrimitiveResult]): Function
                to convert the result of the Braket task to a Qiskit primitive result.
            program_set (ProgramSet): The program set that was run by this task
            index_map (list[list[int]] | None): The per-executable map returned by
                ``ProgramSet.split``. If provided, task results are merged before translation.
        """
        tasks = task if isinstance(task, list) else [task]
        if not tasks:
            raise ValueError("At least one Braket QuantumTask is required")
        job_id = _TASK_ID_DIVIDER.join(task.id for task in tasks)
        super().__init__(job_id=job_id)
        self._tasks = tasks
        self._task = tasks[0]
        self._result_translator = result_translator
        self._program_set = program_set
        self._index_map = index_map
        self._result = None

    @property
    def program_set(self) -> ProgramSet:
        """ProgramSet: The program set that was run by this task"""
        return self._program_set

    @property
    def tasks(self) -> tuple[QuantumTask, ...]:
        """tuple[QuantumTask, ...]: The Braket QuantumTasks created for this primitive job."""
        return tuple(self._tasks)

    def result(self) -> PrimitiveResult:
        if self._result is None:
            task_results = [task.result() for task in self._tasks]
            task_result = (
                ProgramSetQuantumTaskResult.merge(
                    task_results,
                    self._program_set,
                    self._index_map,
                )
                if self._index_map is not None
                else task_results[0]
            )
            self._result = self._result_translator(task_result)
        return self._result

    def status(self) -> JobStatus:
        return self._get_task_status()

    def cancel(self) -> None:
        for task in self._tasks:
            task.cancel()

    def job_id(self) -> str:
        return _TASK_ID_DIVIDER.join(task.id for task in self._tasks)

    def done(self) -> bool:
        return self._get_task_status() == JobStatus.DONE

    def running(self) -> bool:
        return self._get_task_status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        return self._get_task_status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        return self._get_task_status() in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]

    def _get_task_status(self) -> JobStatus:
        statuses = [_TASK_STATUS_MAP[task.state()] for task in self._tasks]
        if JobStatus.ERROR in statuses:
            return JobStatus.ERROR
        if JobStatus.CANCELLED in statuses:
            return JobStatus.CANCELLED
        if not all(status == JobStatus.DONE for status in statuses):
            if JobStatus.RUNNING in statuses:
                return JobStatus.RUNNING
            if JobStatus.INITIALIZING in statuses:
                return JobStatus.INITIALIZING
        return statuses[0]


def run_split_program_set(
    backend: BraketBackend, program_set: ProgramSet, **options: object
) -> tuple[list[QuantumTask], list[list[int]]]:
    """Split and run a program set according to the device executable limit."""
    device = backend._device
    program_sets, index_map = program_set.split(backend._max_program_set_executables)
    if isinstance(device, LocalSimulator):
        tasks = []
        for sub_program_set in program_sets:
            run_options = dict(options)
            if "shots" not in run_options:
                run_options["shots"] = sub_program_set.total_shots
            batch = device.run_batch([sub_program_set], **run_options)
            tasks.extend(
                batch.tasks
                if hasattr(batch, "tasks")
                else [LocalQuantumTask(result) for result in batch.results()]
            )
        return tasks, index_map

    run_options = dict(options)
    if "shots" not in run_options:
        run_options["shots"] = -1
    batch = device.run_batch(program_sets, **run_options)
    return batch.tasks, index_map
