"""Tests for registering control-flow ops on Braket-derived transpiler targets."""

from unittest.mock import Mock

import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler.exceptions import TranspilerError

import qiskit_braket_provider
from braket.aws import AwsDeviceType
from braket.device_schema import DeviceActionType
from braket.devices import LocalSimulator
from braket.ir.openqasm import Program
from qiskit_braket_provider.providers import adapter
from qiskit_braket_provider.providers.compilation import _default_target
from qiskit_braket_provider.providers.target import aws_device_to_target, local_simulator_to_target

from .mocks import mock_iqm_device


def _reset_circuit() -> QuantumCircuit:
    """Excite qubit 1, measure it, flip it back to |0> on a 1, then re-measure it."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.x(1)
    qc.measure(1, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    qc.measure(1, 1)
    return qc


def _simulator_device(*, supports_if: bool) -> Mock:
    properties = LocalSimulator("braket_sv").properties.copy(deep=True)
    action = properties.action[DeviceActionType.OPENQASM]
    if supports_if:
        action.supportedOperations = [*action.supportedOperations, "if"]
    device = Mock()
    device.type = AwsDeviceType.SIMULATOR
    device.name = "sv1"
    device.properties = properties
    return device


@pytest.mark.parametrize("supports_if", [True, False])
def test_qpu_target_if_else_follows_supported_operations(supports_if: bool) -> None:
    target = aws_device_to_target(mock_iqm_device(supports_if=supports_if))

    assert ("if_else" in target.operation_names) is supports_if


@pytest.mark.parametrize("supports_if", [True, False])
def test_simulator_target_if_else_follows_supported_operations(supports_if: bool) -> None:
    device = _simulator_device(supports_if=supports_if)

    assert ("if_else" in aws_device_to_target(device).operation_names) is supports_if
    assert ("if_else" in local_simulator_to_target(device).operation_names) is supports_if


def test_default_target_includes_if_else() -> None:
    assert "if_else" in _default_target([_reset_circuit()]).operation_names


def test_compile_to_oq3_default_target_emits_if_block() -> None:
    oq3 = adapter.compile_to_oq3(_reset_circuit())

    assert "if (b[0]) {" in oq3
    counts = LocalSimulator().run(Program(source=oq3), shots=100).result().measurement_counts
    assert all(outcome[1] == "0" for outcome in counts)


def test_compile_to_oq3_iqm_device_emits_unwrapped_if_block() -> None:
    with pytest.warns(UserWarning, match="control-flow"):
        oq3 = adapter.compile_to_oq3(
            _reset_circuit(), braket_device=mock_iqm_device(), optimization_level=1
        )

    assert "if (b[0]) {\nprx(" in oq3
    assert "#pragma braket verbatim" not in oq3
    assert "b[1] = measure $" in oq3
    assert "h " not in oq3
    assert "x " not in oq3


def test_compile_to_oq3_iqm_device_without_if_support_raises() -> None:
    with pytest.raises(TranspilerError, match="if_else"):
        adapter.compile_to_oq3(_reset_circuit(), braket_device=mock_iqm_device(supports_if=False))


def test_compile_to_oq3_and_to_oq3_exported_from_package_root() -> None:
    assert qiskit_braket_provider.compile_to_oq3 is adapter.compile_to_oq3
    assert qiskit_braket_provider.to_oq3 is adapter.to_oq3
