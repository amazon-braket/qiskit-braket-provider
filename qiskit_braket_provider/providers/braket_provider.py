"""Amazon Braket provider."""

import warnings

from qiskit.providers.exceptions import QiskitBackendNotFoundError

from braket.aws import AwsDevice, AwsDeviceType
from braket.device_schema.dwave import DwaveDeviceCapabilities
from braket.device_schema.quera import QueraDeviceCapabilities
from braket.device_schema.xanadu import XanaduDeviceCapabilities

from .braket_backend import BraketAwsBackend, BraketLocalBackend


def _is_aws_simulator(device: AwsDevice) -> bool:
    return device.type in (AwsDeviceType.SIMULATOR, AwsDeviceType.SIMULATOR.value)


def _is_supported_gate_model(device: AwsDevice) -> bool:
    return not isinstance(
        device.properties,
        (
            DwaveDeviceCapabilities,
            XanaduDeviceCapabilities,
            QueraDeviceCapabilities,
        ),
    )


class BraketProvider:
    """Provides access to Amazon Braket backends.

    Example:
        >>> provider = BraketProvider()
        >>> backends = provider.backends()
        >>> backends
        [BraketBackend[Aria 1],
         BraketBackend[Aria 2],
         BraketBackend[Aspen-M-3],
         BraketBackend[Forte 1],
         BraketBackend[Harmony],
         BraketBackend[Lucy],
         BraketBackend[SV1],
         BraketBackend[TN1],
         BraketBackend[dm1]]
    """

    def get_backend(
        self, name: str | None = None, **kwargs
    ) -> BraketAwsBackend | BraketLocalBackend:
        """Return a single backend matching the specified filters.

        Args:
            name (str): name of the selected backend
            **kwargs: dict with additional options for filtering and storing aws session.
                Set ``emulator=True`` to return a local emulator backend for an AWS device.
        Returns:
            BraketAwsBackend | BraketLocalBackend: a backend matching the filters.
        Raises:
            QiskitBackendNotFoundError: if no backend could be found or
            more than one backend matches the filters.
        """
        backends = self.backends(name=name, **kwargs)
        if len(backends) > 1:
            raise QiskitBackendNotFoundError("More than one backend matches the criteria")
        if not backends:
            raise QiskitBackendNotFoundError("No backend matches the criteria")
        return backends[0]

    def backends(
        self,
        name: str | None = None,
        **kwargs,
    ) -> list[BraketAwsBackend | BraketLocalBackend]:
        """Return a list of backends matching the specified filters.

        Args:
            name (str): name of the selected backend
            **kwargs: dict with additional options for filtering and storing aws session.
                Set ``emulator=True`` to return local emulator backends for AWS devices.
        Returns:
            BraketAwsBackend | BraketLocalBackend: a list of backends matching the filters.
        """
        emulator = kwargs.pop("emulator", False)
        if kwargs.get("local"):
            return [
                BraketLocalBackend(name="braket_sv"),
                BraketLocalBackend(name="braket_dm"),
            ]
        names = [name] if name else None
        devices = AwsDevice.get_devices(names=names, **kwargs)
        # filter by supported devices
        # gate models are only supported
        supported_gate_model_devices = [d for d in devices if _is_supported_gate_model(d)]
        supported_devices = [
            d for d in supported_gate_model_devices if not emulator or not _is_aws_simulator(d)
        ]
        if emulator and name and not supported_devices:
            simulator_matches = [d for d in supported_gate_model_devices if _is_aws_simulator(d)]
            if simulator_matches:
                raise QiskitBackendNotFoundError(
                    f"Backend {name} does not support device emulation; emulators are available "
                    "only for supported QPU devices."
                )
        backends = [
            BraketAwsBackend(
                device=device,
                provider=self,
                name=device.name,
                description=f"AWS Device: {device.provider_name} {device.name}.",
                online_date=device.properties.service.updatedAt,
                backend_version="2",
            )
            for device in supported_devices
        ]
        return [backend.emulator() for backend in backends] if emulator else backends


class AWSBraketProvider(BraketProvider):
    """AWSBraketProvider class for accessing Amazon Braket backends."""

    def __init_subclass__(cls, **kwargs) -> None:
        """This throws a deprecation warning on subclassing."""
        warnings.warn(f"{cls.__name__} is deprecated.", DeprecationWarning, stacklevel=2)
        super().__init_subclass__(**kwargs)

    def __init__(self) -> None:
        """This throws a deprecation warning on initialization."""
        warnings.warn(
            f"{self.__class__.__name__} is deprecated. Use BraketProvider instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
