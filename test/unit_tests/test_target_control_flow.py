"""Tests for registering if_else on Braket-derived transpiler targets."""

from collections.abc import Callable
from unittest.mock import Mock

import pytest
from qiskit.transpiler import Target

from qiskit_braket_provider.providers.target import aws_device_to_target, local_simulator_to_target

from .mocks import mock_iqm_device, mock_simulator_device


@pytest.mark.parametrize("supports_if", [True, False])
@pytest.mark.parametrize(
    ("make_device", "to_target"),
    [
        (mock_iqm_device, aws_device_to_target),
        (mock_simulator_device, aws_device_to_target),
        (mock_simulator_device, local_simulator_to_target),
    ],
    ids=["qpu", "simulator", "local_simulator"],
)
def test_target_if_else_follows_supported_operations(
    make_device: Callable[..., Mock], to_target: Callable[[Mock], Target], supports_if: bool
) -> None:
    """A device target registers if_else exactly when the device advertises ``if``."""
    target = to_target(make_device(supports_if=supports_if))

    assert ("if_else" in target.operation_names) is supports_if
