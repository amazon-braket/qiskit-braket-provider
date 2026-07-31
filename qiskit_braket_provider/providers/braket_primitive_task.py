import warnings
from collections.abc import Callable

from qiskit.primitives import BasePrimitiveJob, PrimitiveResult, PubResult
from qiskit.providers import JobStatus

from braket.devices import LocalSimulator
from braket.emulation import Emulator
from braket.program_sets import ProgramSet
from braket.tasks import ProgramSetQuantumTaskResult, QuantumTask
from qiskit_braket_provider.providers.braket_backend import BraketBackend
from qiskit_braket_provider.providers.braket_quantum_task import _aggregate_task_status

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
            try:
                task.cancel()
            except Exception as ex:  # ruff: ignore[blind-except]
                warnings.warn(
                    f"Failed to cancel Braket task {task.id}: {ex}",
                    stacklevel=2,
                )

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
        return _aggregate_task_status({task.state() for task in self._tasks})


def run_split_program_set(
    backend: BraketBackend, program_set: ProgramSet, **options: object
) -> tuple[list[QuantumTask], list[list[int]] | None]:
    """Split and run a program set according to the device executable limit."""
    device = backend._device
    run_options = {**options, "shots": None}
    if isinstance(device, LocalSimulator):
        return [device.run(program_set, **run_options)], None

    program_sets, index_map = program_set.split(backend._max_program_set_executables)
    if isinstance(device, Emulator):
        tasks = [device.run(sub_program_set, **run_options) for sub_program_set in program_sets]
        return tasks, index_map

    batch = device.run_batch(program_sets, **{**options, "shots": -1})
    return batch.tasks, index_map
