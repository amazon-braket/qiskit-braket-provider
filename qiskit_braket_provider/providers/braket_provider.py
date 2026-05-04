"""Amazon Braket provider."""

import warnings

from qiskit.providers.exceptions import QiskitBackendNotFoundError

from braket.aws import AwsDevice
from braket.device_schema.dwave import DwaveDeviceCapabilities
from braket.device_schema.quera import QueraDeviceCapabilities
from braket.device_schema.xanadu import XanaduDeviceCapabilities

from .braket_backend import BraketAwsBackend, BraketLocalBackend


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

    def get_backend(self, name: str | None = None, **kwargs) -> BraketAwsBackend:
        """Return a single backend matching the specified filters.

        Args:
            name (str): name of the selected backend
            **kwargs: dict with additional options for filtering and storing aws session
        Returns:
            BraketAwsBackend: a backend matching the filters.
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
