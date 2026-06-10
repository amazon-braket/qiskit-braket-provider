"""Amazon Braket provider."""

import warnings

from qiskit.providers.exceptions import QiskitBackendNotFoundError

from braket.aws import AwsDevice, AwsDeviceType
from braket.device_schema.dwave import DwaveDeviceCapabilities
from braket.device_schema.quera import QueraDeviceCapabilities
from braket.device_schema.xanadu import XanaduDeviceCapabilities

from .braket_backend import BraketAwsBackend, BraketEmulatorBackend, BraketLocalBackend


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
        self, name: str | None = None, emulator: bool = False, **kwargs
    ) -> BraketAwsBackend | BraketEmulatorBackend:
        """Return a single backend matching the specified filters.

        Args:
            name (str): name of the selected backend
            emulator (bool): return a local emulator backend for the selected device
                instead of the device itself. Default: ``False``.
            **kwargs: dict with additional options for filtering and storing aws session
        Returns:
            BraketAwsBackend | BraketEmulatorBackend: a backend matching the filters.
        Raises:
            QiskitBackendNotFoundError: if no backend could be found or
            more than one backend matches the filters.
        """
        backends = self.backends(name=name, emulator=emulator, **kwargs)
        if len(backends) > 1:
            raise QiskitBackendNotFoundError("More than one backend matches the criteria")
        if not backends:
            raise QiskitBackendNotFoundError("No backend matches the criteria")
        return backends[0]

    def backends(
        self,
        name: str | None = None,
        emulator: bool = False,
        **kwargs,
    ) -> list[BraketAwsBackend | BraketLocalBackend | BraketEmulatorBackend]:
        """Return a list of backends matching the specified filters.

        Args:
            name (str): name of the selected backend
            emulator (bool): return local emulator backends for the matching devices
                instead of the devices themselves. Emulators mimic the gate set,
                connectivity and noise of a device while executing locally. Only QPUs
                have emulators, so managed simulators are excluded. Default: ``False``.
            **kwargs: dict with additional options for filtering and storing aws session
        Returns:
            BraketAwsBackend: a list of backends matching the filters.
        """
        if kwargs.get("local"):
            return [
                BraketLocalBackend(name="braket_sv"),
                BraketLocalBackend(name="braket_dm"),
            ]
        names = [name] if name else None
        devices = AwsDevice.get_devices(names=names, **kwargs)
        # filter by supported devices
        # gate models are only supported
        supported_devices = [
            d
            for d in devices
            if not isinstance(
                d.properties,
                (
                    DwaveDeviceCapabilities,
                    XanaduDeviceCapabilities,
                    QueraDeviceCapabilities,
                ),
            )
        ]
        if emulator:
            return [
                BraketEmulatorBackend(
                    device=device,
                    provider=self,
                    name=device.name,
                    description=f"Emulator for AWS Device: {device.provider_name} {device.name}.",
                    online_date=device.properties.service.updatedAt,
                    backend_version="2",
                )
                for device in supported_devices
                if device.type == AwsDeviceType.QPU
            ]
        return [
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
