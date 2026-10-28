"""Tests that QPU targets only use calibration data for qubits and edges in the device topology."""

import warnings
from collections.abc import Iterable
from unittest.mock import Mock

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.exceptions import TranspilerError

from braket.aws import AwsDevice
from braket.device_schema import StandardizedGateModelQpuDeviceProperties
from qiskit_braket_provider.providers.target import aws_device_to_target
from test.unit_tests.mocks import mock_iqm_device

_LINEAR_EDGES = [(1, 2), (2, 3), (3, 4), (4, 5)]


def _standardized_properties(
    qubits: Iterable[int], edges: Iterable[tuple[int, int]]
) -> StandardizedGateModelQpuDeviceProperties:
    return StandardizedGateModelQpuDeviceProperties.parse_obj({
        "braketSchemaHeader": {
            "name": "braket.device_schema.standardized_gate_model_qpu_device_properties",
            "version": "1",
        },
        "oneQubitProperties": {
            str(q): {
                "T1": {"value": 30.0, "unit": "us"},
                "T2": {"value": 40.0, "unit": "us"},
                "oneQubitFidelity": [
                    {"fidelityType": {"name": "RANDOMIZED_BENCHMARKING"}, "fidelity": 0.999},
                    {"fidelityType": {"name": "READOUT"}, "fidelity": 0.95},
                ],
            }
            for q in qubits
        },
        "twoQubitProperties": {
            f"{u}-{v}": {
                "twoQubitGateFidelity": [
                    {
                        "direction": {"control": u, "target": v},
                        "gateName": "CZ",
                        "fidelity": 0.99,
                        "fidelityType": {"name": "INTERLEAVED_RANDOMIZED_BENCHMARKING"},
                    }
                ]
            }
            for u, v in edges
        },
    })


def _iqm_device(
    connectivity_graph: dict[str, list[str]] | None,
    calibrated_edges: Iterable[tuple[int, int]],
) -> Mock:
    """Mock IQM device on qubits 1-5 whose topology is built the same way ``AwsDevice`` does."""
    device = mock_iqm_device()
    if connectivity_graph is not None:
        device.properties.paradigm.connectivity.connectivityGraph = connectivity_graph
    device.properties.standardized = _standardized_properties(range(1, 6), calibrated_edges)
    device.topology_graph = AwsDevice._construct_topology_graph(device)
    return device


def test_calibrated_edge_missing_from_topology_is_dropped():
    """A calibrated edge absent from connectivityGraph must not become a usable target edge."""
    device = _iqm_device(None, [*_LINEAR_EDGES, (2, 4)])

    with pytest.warns(UserWarning, match="Edge 2-4 not found in topology"):
        target = aws_device_to_target(device)

    # Qubits 1-5 map to target indices 0-4
    expected = {(u - 1, v - 1) for u, v in _LINEAR_EDGES}
    assert set(target["cz"]) == expected | {(v, u) for u, v in expected}
    assert target["cz"][0, 1].error == pytest.approx(0.01)


def test_compiled_circuit_only_uses_topology_edges():
    device = _iqm_device(None, [*_LINEAR_EDGES, (2, 4)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        target = aws_device_to_target(device)
    circuit = QuantumCircuit(5)
    circuit.cz(1, 3)

    compiled = transpile(circuit, target=target, initial_layout=list(range(5)), seed_transpiler=0)

    topology_edges = {(u - 1, v - 1) for u, v in _LINEAR_EDGES}
    for instruction in compiled.data:
        if instruction.operation.num_qubits == 2:
            u, v = (compiled.find_bit(q).index for q in instruction.qubits)
            assert (u, v) in topology_edges or (v, u) in topology_edges


def test_empty_connectivity_graph_does_not_use_calibration_data():
    """Calibration data must not stand in for an empty connectivityGraph."""
    device = _iqm_device({}, _LINEAR_EDGES)

    with pytest.warns(UserWarning, match="not in topology"):
        target = aws_device_to_target(device)

    assert target.num_qubits == 0
    assert not target["cz"]
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)
    with pytest.raises(TranspilerError):
        transpile(circuit, target=target)
