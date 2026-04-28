"""Amazon Braket backends."""

from __future__ import annotations

import copy
import enum
import logging
import warnings
from abc import ABC
from typing import TYPE_CHECKING, Any, Generic, NoReturn, TypeVar

from qiskit import QuantumCircuit
from qiskit.providers import BackendV2, Options, QubitProperties
from qiskit.transpiler import PassManager, Target

from braket.aws import AwsDevice, AwsDeviceType, AwsQuantumTask
from braket.aws.queue_information import QueueDepthInfo
from braket.circuits import Circuit
from braket.device_schema import DeviceActionType
from braket.devices import Device, LocalSimulator
from braket.program_sets import ProgramSet
from braket.tasks.local_quantum_task import LocalQuantumTask

from .._version import __version__
from ..exception import QiskitBraketException
from .adapter import (
    aws_device_to_target,
    gateset_from_properties,
    local_simulator_to_target,
    native_angle_restrictions,
    native_gate_set,
    to_braket,
)
from .braket_quantum_task import BraketQuantumTask

if TYPE_CHECKING:
    import datetime
    from collections.abc import Callable, Iterable

    from .braket_provider import BraketProvider

logger = logging.getLogger(__name__)

_TASK_ID_DIVIDER = ";"

T = TypeVar("T", bound=Device, covariant=True)  # noqa: PLC0105


class BraketBackend(BackendV2, ABC, Generic[T]):
    """Base Qiskit backend for Amazon Braket devices."""

    def __init__(self, device: T, name: str, **fields: Any) -> None:  # noqa: ANN401
        super().__init__(name=name, **fields)
        self._device = device
        self._supports_program_sets = (
            DeviceActionType.OPENQASM_PROGRAM_SET in self._device.properties.action
        )

    def __repr__(self) -> str:
        return f"BraketBackend[{self.name}]"

    @property
    def qubit_labels(self) -> tuple[int, ...] | None:
        """
        tuple[int, ...] | None: The qubit labels of the underlying device, in ascending order.

        Unlike the qubits in the target, these labels are not necessarily contiguous.
        """
        return None

    def _validate_meas_level(self, meas_level: enum.Enum | int) -> None:
        if isinstance(meas_level, enum.Enum):
            meas_level = meas_level.value
        if meas_level != 2:
            raise QiskitBraketException(
                f"Device {self.name} only supports classified measurement "
                f"results, received meas_level={meas_level}."
            )

    def get_gateset(self, native: bool = False) -> set[str] | None:
        """Get the gate set of the device.

        Args:
            native (bool): Whether to return the device's native gates. Default: ``False``.

        Returns:
            set[str] | None: The requested gate set.
        """
        if native:
            return native_gate_set(self._device.properties)
        action = self._device.properties.action.get(DeviceActionType.OPENQASM)
        return gateset_from_properties(action) if action else None

    def _run_program_set(
        self, braket_circuits: list[Circuit], shots: int | None, **options: Any  # noqa: ANN401
    ) -> BraketQuantumTask:
        program_set = ProgramSet(braket_circuits, shots_per_executable=shots)
        task = self._device.run(program_set, **options)
        return BraketQuantumTask(
            task_id=task.id, tasks=task, backend=self, shots=program_set.total_shots
        )


class BraketLocalBackend(BraketBackend[LocalSimulator]):
    """Runs quantum circuits on the Braket local simulator."""

    def __init__(self, name: str = "default", **fields: Any) -> None:  # noqa: ANN401
        """Initialize the backend.

        Example:
            >>> device = LocalSimulator()                    #Local State Vector Simulator
            >>> device = LocalSimulator("default")           #Local State Vector Simulator
            >>> device = LocalSimulator(name="default")      #Local State Vector Simulator
            >>> device = LocalSimulator(name="braket_sv")    #Local State Vector Simulator
            >>> device = LocalSimulator(name="braket_dm")    #Local Density Matrix Simulator

        Args:
            name (str): Name of backend. Default: ``default``.
            **fields: Extra arguments.
        """
        simulator = LocalSimulator(backend=name)
        super().__init__(simulator, name or simulator.name, **fields)
        self._target = local_simulator_to_target(self._device)
        self._gateset = self.get_gateset()
        self.status = self._device.status

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> None:
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    @property
    def dtm(self) -> float:
        raise NotImplementedError(
            f"System time resolution of output signals is not supported by {self.name}."
        )

    @property
    def meas_map(self) -> list[list[int]]:
        raise NotImplementedError(f"Measurement map is not supported by {self.name}.")

    def qubit_properties(self, qubit: int | list[int]) -> QubitProperties | list[QubitProperties]:
        raise NotImplementedError

    def drive_channel(self, qubit: int) -> NoReturn:
        raise NotImplementedError(f"Drive channel is not supported by {self.name}.")

    def measure_channel(self, qubit: int) -> NoReturn:
        raise NotImplementedError(f"Measure channel is not supported by {self.name}.")

    def acquire_channel(self, qubit: int) -> NoReturn:
        raise NotImplementedError(f"Acquire channel is not supported by {self.name}.")

    def control_channel(self, qubits: Iterable[int]) -> NoReturn:
        raise NotImplementedError(f"Control channel is not supported by {self.name}.")

    def run(
        self,
        run_input: QuantumCircuit | list[QuantumCircuit],
        *,
        shots: int = 1024,
        **options: Any,  # noqa: ANN401
    ) -> BraketQuantumTask:
        convert_input = [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        verbatim = options.pop("verbatim", False)
        circuits: list[Circuit] = [
            to_braket(circ, target=self._target if not verbatim else None, verbatim=verbatim)
            for circ in convert_input
        ]

        if shots == 0:
            circuits = [x.state_vector() for x in circuits]
        if "meas_level" in options:
            self._validate_meas_level(options["meas_level"])
            del options["meas_level"]
        tasks = []
        try:
            for circuit in circuits:
                task: LocalQuantumTask = self._device.run(task_specification=circuit, shots=shots)
                tasks.append(task)

        except Exception as ex:
            logger.error("During creation of tasks an error occurred: %s", ex)
            logger.error("Cancelling all tasks %d!", len(tasks))
            for task in tasks:
                logger.error("Attempt to cancel %s...", task.id)
                task.cancel()
                logger.error("State of %s: %s.", task.id, task.state())
            raise

        task_id = _TASK_ID_DIVIDER.join(task.id for task in tasks)

        return BraketQuantumTask(
            task_id=task_id,
            tasks=tasks,
            backend=self,
            shots=shots,
        )


class BraketAwsBackend(BraketBackend[AwsDevice]):
    """Runs quantum circuits on the Amazon Braket service."""

    def __init__(
        self,
        arn: str | None = None,
        provider: BraketProvider | None = None,
        name: str | None = None,
        description: str | None = None,
        online_date: datetime.datetime | None = None,
        backend_version: str | None = None,
        *,
        device: AwsDevice | None = None,
        **fields: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the backend.

        Example:
            >>> provider = BraketProvider()
            >>> backend = provider.get_backend("SV1")
            >>> transpiled_circuit = transpile(circuit, backend=backend)
            >>> backend.run(transpiled_circuit, shots=10).result().get_counts()
            {"100": 10, "001": 10}

        Args:
            arn (str | None): ARN of the Braket device. Default: ``None``.
            provider: Qiskit provider for this backend. Default: ``None``.
            name (str | None): Name of backend. Default: ``None``.
            description (str | None): Description of backend. Default: ``None``.
            online_date (datetime | None): Online date. Default: ``None``.
            backend_version (str | None): Backend version. Default: ``None``.
            device (AwsDevice | None): Braket device instance. Default: ``None``.
            **fields: Extra arguments.
        """
        if not (arn or device):
            raise ValueError("Must specify either arn or device")
        if arn and device:
            raise ValueError("Can only specify one of arn and device")
        aws_device = AwsDevice(arn) if arn else device
        super().__init__(
            aws_device,
            name or aws_device.name,
            provider=provider,
            description=description,
            online_date=online_date,
            backend_version=backend_version,
            **fields,
        )
        self._device.aws_session.add_braket_user_agent(f"QiskitBraketProvider/{__version__}")
        self._target = aws_device_to_target(device=self._device)
        self._qubit_labels = (
            tuple(sorted(self._device.topology_graph.nodes))
            if self._device.topology_graph
            else None
        )
        self._gateset = self.get_gateset()

    def retrieve_job(self, task_id: str) -> BraketQuantumTask:
        """Return a single job submitted to AWS backend.

        Args:
            task_id (str): ID of the task to retrieve.

        Returns:
            The job with the given ID.
        """
        task_ids = task_id.split(_TASK_ID_DIVIDER)

        return BraketQuantumTask(
            task_id=task_id,
            backend=self,
            tasks=[AwsQuantumTask(arn=task_id) for task_id in task_ids],
        )

    @property
    def target(self) -> Target:
        return self._target

    @property
    def max_circuits(self) -> None:
        return None

    @property
    def qubit_labels(self) -> tuple[int, ...] | None:
        return self._qubit_labels

    @classmethod
    def _default_options(cls) -> Options:
        return Options()

    def qubit_properties(self, qubit: int | list[int]) -> QubitProperties | list[QubitProperties]:
        # TODO: fetch information from device.properties.provider
        raise NotImplementedError

    def queue_depth(self) -> QueueDepthInfo:
        """
        Task queue depth refers to the total number of quantum tasks currently waiting
        to run on a particular device.

        Returns:
            QueueDepthInfo: Instance of the QueueDepth class representing queue depth
            information for quantum jobs and hybrid jobs.
            Queue depth refers to the number of quantum jobs and hybrid jobs queued on a particular
            device. The normal tasks refers to the quantum jobs not submitted via Hybrid Jobs.
            Whereas, the priority tasks refers to the total number of quantum jobs waiting to run
            submitted through Amazon Braket Hybrid Jobs. These tasks run before the normal tasks.
            If the queue depth for normal or priority quantum tasks is greater than 4000, we display
            their respective queue depth as '>4000'. Similarly, for hybrid jobs if there are more
            than 1000 jobs queued on a device, display the hybrid jobs queue depth as '>1000'.
            Additionally, for QPUs if hybrid jobs queue depth is 0, we display information about
            priority and count of the running hybrid job.

        Example:
            Queue depth information for a running hybrid job.
            >>> device = BraketProvider().get_backend("SV1")
            >>> print(device.queue_depth())
            QueueDepthInfo(quantum_tasks={<QueueType.NORMAL: 'Normal'>: '0',
            <QueueType.PRIORITY: 'Priority'>: '1'}, jobs='0 (1 prioritized job(s) running)')

            If more than 4000 quantum jobs queued on a device.
            >>> device = BraketProvider().get_backend("SV1")
            >>> print(device.queue_depth())
            QueueDepthInfo(quantum_tasks={<QueueType.NORMAL: 'Normal'>: '>4000',
            <QueueType.PRIORITY: 'Priority'>: '2000'}, jobs='100')
        """
        return self._device.queue_depth()

    @property
    def dtm(self) -> float:
        raise NotImplementedError(
            f"System time resolution of output signals is not supported by {self.name}."
        )

    @property
    def meas_map(self) -> list[list[int]]:
        raise NotImplementedError(f"Measurement map is not supported by {self.name}.")

    def drive_channel(self, qubit: int) -> NoReturn:
        raise NotImplementedError(f"Drive channel is not supported by {self.name}.")

    def measure_channel(self, qubit: int) -> NoReturn:
        raise NotImplementedError(f"Measure channel is not supported by {self.name}.")

    def acquire_channel(self, qubit: int) -> NoReturn:
        raise NotImplementedError(f"Acquire channel is not supported by {self.name}.")

    def control_channel(self, qubits: Iterable[int]) -> NoReturn:
        raise NotImplementedError(f"Control channel is not supported by {self.name}.")

    def run(
        self,
        run_input: QuantumCircuit | list[QuantumCircuit],
        shots: int | None = None,
        verbatim: bool = False,
        native: bool = False,
        *,
        optimization_level: int = 0,
        callback: Callable | None = None,
        num_processes: int | None = None,
        pass_manager: PassManager | None = None,
        **options: Any,  # noqa: ANN401
    ) -> BraketQuantumTask:
        """Execute ``QuantumCircuit``s on a ``BraketAwsBackend``

        Args:
            run_input (QuantumCircuit | list[QuantumCircuit]): ``QuantumCircuit`` or list
                of ``QuantumCircuit``s.
            shots (int | None): Number of measurement repetitions for the BraketAwsBackend.
                Default: ``None``.
            verbatim (bool): Submit as a verbatim circuit (i.e. no transpilation).
                Default: ``False``.
            native (bool): use the Qiskit transpiler to compile to a verbatim native circuit
                with noise aware transpilation when available and optimization_level > 2.
                Default: ``False``.
            optimization_level (int): Qiskit transpiler optimization level (see adapter.py).
                Default: 0.
            callback (Callable | None): Function for the Qiskit transpiler. Default: ``None``.
            num_processes (int | None): Allow for parallel transpilation. Default: ``None``.
            pass_manager (PassManager | None): User-specified ``PassManager`` for the
                Qiskit transpiler (creates verbatim circuits). Default: ``None``.
        """

        if isinstance(run_input, QuantumCircuit):
            circuits = [run_input]
        elif isinstance(run_input, list):
            circuits = run_input
        else:
            raise QiskitBraketException(f"Unsupported input type: {type(run_input)}")

        if "meas_level" in options:
            self._validate_meas_level(options["meas_level"])
            del options["meas_level"]

        # Always use target for simulator
        target, basis_gates = self._target_and_basis_gates(native, pass_manager)
        braket_circuits = (
            to_braket(circuits, qubit_labels=self._qubit_labels, verbatim=True)
            if verbatim
            else to_braket(
                circuits,
                qubit_labels=self._qubit_labels,
                target=target,
                basis_gates=basis_gates,
                angle_restrictions=(
                    native_angle_restrictions(self._device.properties) if native else None
                ),
                optimization_level=optimization_level,
                callback=callback,
                num_processes=num_processes,
                pass_manager=pass_manager,
            )
        )
        return (
            self._run_program_set(braket_circuits, shots, **options)
            if self._supports_program_sets and shots != 0 and len(braket_circuits) > 1
            else self._run_batch(braket_circuits, shots, **options)
        )

    def _target_and_basis_gates(
        self, native: bool, pass_manager: PassManager
    ) -> tuple[Target | None, set[str] | None]:
        if pass_manager:
            return None, None
        if native or self._device.type == AwsDeviceType.SIMULATOR:
            # Always use target for simulator
            return self._target, None
        return None, self._gateset

    def _run_batch(
        self, braket_circuits: list[Circuit], shots: int, **options: Any  # noqa: ANN401
    ) -> BraketQuantumTask:
        batch_task = self._device.run_batch(braket_circuits, shots=shots, **options)
        tasks: list[AwsQuantumTask] = batch_task.tasks
        task_id = _TASK_ID_DIVIDER.join(task.id for task in tasks)
        return BraketQuantumTask(task_id=task_id, tasks=tasks, backend=self, shots=shots)

    def __deepcopy__(self, memo: dict[int, Any]) -> BraketAwsBackend:
        """Create deepcopy of the BraketBackend.

        Note: the underlying self._device, and thus self._device.aws_session
        is shared between copies.
        """
        result = copy.copy(self)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key != "_device":
                setattr(result, key, copy.deepcopy(value, memo))  # Pass memo along
        return result


class AWSBraketBackend(BraketAwsBackend):
    """AWSBraketBackend."""

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """This throws a deprecation warning on subclassing."""
        warnings.warn(f"{cls.__name__} is deprecated.", DeprecationWarning, stacklevel=2)
        super().__init_subclass__(**kwargs)

    def __init__(
        self,
        device: AwsDevice,
        provider: BraketProvider | None = None,
        name: str | None = None,
        description: str | None = None,
        online_date: datetime.datetime | None = None,
        backend_version: str | None = None,
        **fields: Any,  # noqa: ANN401
    ) -> None:
        """This throws a deprecation warning on initialization."""
        warnings.warn(
            f"{self.__class__.__name__} is deprecated. Use BraketAwsBackend instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            device=device,
            provider=provider,
            name=name,
            description=description,
            online_date=online_date,
            backend_version=backend_version,
            **fields,
        )
