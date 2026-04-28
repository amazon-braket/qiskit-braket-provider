"""Tests for Amazon Braket provider."""

import uuid
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from networkx import DiGraph, from_dict_of_lists, relabel_nodes
from qiskit import QuantumCircuit
from qiskit import circuit as qiskit_circuit
from qiskit.compiler import transpile
from qiskit.providers.exceptions import QiskitBackendNotFoundError

from braket.aws import AwsDevice, AwsDeviceType, AwsQuantumTaskBatch, AwsSession
from braket.aws.queue_information import QuantumTaskQueueInfo, QueueType
from braket.circuits import Circuit
from qiskit_braket_provider import (
    AWSBraketProvider,
    BraketAwsBackend,
    BraketLocalBackend,
    BraketProvider,
    to_braket,
    to_qiskit,
)
from qiskit_braket_provider.providers.braket_backend import BraketBackend
from test.unit_tests.mocks import (
    MOCK_GATE_MODEL_SIMULATOR_SV,
    MOCK_GATE_MODEL_SIMULATOR_TN,
    MOCK_RIGETTI_GATE_MODEL_M_3_QPU,
    MOCK_RIGETTI_M_3_QPU_CAPABILITIES,
    SIMULATOR_REGION,
)


class TestBraketProvider(TestCase):
    """Tests BraketProvider."""

    def setUp(self) -> None:
        self.mock_session = Mock()
        simulators = [MOCK_GATE_MODEL_SIMULATOR_SV, MOCK_GATE_MODEL_SIMULATOR_TN]
        self.mock_session.get_device.side_effect = simulators
        self.mock_session.region = SIMULATOR_REGION
        self.mock_session.boto_session.region_name = SIMULATOR_REGION
        self.mock_session.search_devices.return_value = simulators

        self.empty_mock_session = Mock()
        self.empty_mock_session.get_device.side_effect = []
        self.empty_mock_session.region = SIMULATOR_REGION
        self.empty_mock_session.boto_session.region_name = SIMULATOR_REGION
        self.empty_mock_session.search_devices.return_value = []

    def test_provider_backend(self) -> None:
        """Test QiskitBackendNotFoundError is raised"""
        provider = BraketProvider()

        # Matches QiskitBackendNotFoundError where multiple backends are found
        with self.assertRaises(QiskitBackendNotFoundError) as err1:
            provider.get_backend(aws_session=self.mock_session, types=[AwsDeviceType.SIMULATOR])
        self.assertIsInstance(err1.exception, QiskitBackendNotFoundError)

        # Matches QiskitBackendNotFoundError where no backends are found
        with self.assertRaises(QiskitBackendNotFoundError) as err2:
            provider.get_backend(
                aws_session=self.empty_mock_session, types=[AwsDeviceType.SIMULATOR]
            )
        self.assertIsInstance(err2.exception, QiskitBackendNotFoundError)

    def test_provider_backends(self) -> None:
        """Tests provider."""
        provider = BraketProvider()
        backends = provider.backends(aws_session=self.mock_session, types=[AwsDeviceType.SIMULATOR])

        self.assertTrue(len(backends) > 0)
        for backend in backends:
            with self.subTest(f"{backend.name}"):
                self.assertIsInstance(backend, BraketBackend)

    def test_deprecation_warning_on_init(self) -> None:
        """Check if a DeprecationWarning is raised when AWSBraketProvider is initialized"""
        with self.assertWarns(DeprecationWarning):
            AWSBraketProvider()

    def test_deprecation_warning_on_subclass(self) -> None:
        """Check if a DeprecationWarning is raised when a subclass of AWSBraketProvider is created"""
        with self.assertWarns(DeprecationWarning):

            class SubclassAWSBraketProvider(AWSBraketProvider):
                """This is a subclass of AWSBraketProvider for testing purposes."""

    def test_provider_backends_kwargs_local(self) -> None:
        """Tests getting local backends using kwargs"""
        provider = BraketProvider()

        self.assertIsInstance(provider.backends(name=None, local="sv1")[0], BraketLocalBackend)

    def test_real_devices(self) -> None:
        """Tests real devices."""
        with patch(
            "qiskit_braket_provider.providers.braket_provider.AwsDevice"
        ) as mock_get_devices:
            mock_get_devices.get_devices.return_value = [
                AwsDevice(MOCK_GATE_MODEL_SIMULATOR_SV["deviceArn"], self.mock_session),
                AwsDevice(MOCK_GATE_MODEL_SIMULATOR_TN["deviceArn"], self.mock_session),
            ]
            provider = BraketProvider()
            backends = provider.backends()
            self.assertTrue(len(backends) > 0)
            for backend in backends:
                with self.subTest(f"{backend.name}"):
                    self.assertIsInstance(backend, BraketAwsBackend)

            online_simulators_backends = provider.backends(statuses=["ONLINE"], types=["SIMULATOR"])
            for backend in online_simulators_backends:
                with self.subTest(f"{backend.name}"):
                    self.assertIsInstance(backend, BraketAwsBackend)

    @patch("qiskit_braket_provider.providers.braket_backend.BraketAwsBackend")
    @patch("qiskit_braket_provider.providers.braket_backend.AwsDevice.get_devices")
    def test_qiskit_circuit_transpilation_run(
        self, mock_get_devices: MagicMock, mock_aws_braket_backend: MagicMock
    ) -> None:
        """Tests qiskit circuit transpilation."""
        mock_get_devices.return_value = [
            AwsDevice(MOCK_GATE_MODEL_SIMULATOR_SV["deviceArn"], self.mock_session)
        ]
        s3_target = AwsSession.S3DestinationFolder("mock_bucket", "mock_key")
        q_circuit = qiskit_circuit.QuantumCircuit(2)
        q_circuit.h(0)
        q_circuit.cx(0, 1)
        braket_circuit = Circuit().h(0).cnot(0, 1)

        mock_aws_braket_backend = Mock(spec=BraketAwsBackend)
        mock_aws_braket_backend._device = Mock(spec=AwsDevice)
        task = AwsQuantumTaskBatch(
            Mock(),
            MOCK_GATE_MODEL_SIMULATOR_SV["deviceArn"],
            braket_circuit,
            s3_target,
            1000,
            max_parallel=10,
        )
        task_mock = Mock()
        task_mock.id = str(uuid.uuid4())
        task_mock.state.return_value = "RUNNING"
        task = Mock(spec=AwsQuantumTaskBatch, return_value=task)
        task.tasks = [task_mock]

        provider = BraketProvider()
        state_vector_backend = provider.get_backend("SV1", aws_session=self.mock_session)
        transpiled_circuit = transpile(q_circuit, backend=state_vector_backend, seed_transpiler=42)

        state_vector_backend._device.run_batch = Mock(spec=AwsQuantumTaskBatch, return_value=task)
        result = state_vector_backend.run(transpiled_circuit, shots=10)
        self.assertTrue(result)

    @patch("braket.aws.aws_device.AwsDevice.get_devices")
    def test_noncontigous_qubit_indices_qiskit_transpilation(
        self, mock_get_devices: MagicMock
    ) -> None:
        """Tests circuit transpilation with noncontiguous qubit indices."""

        mock_m_3_device = Mock()
        mock_m_3_device.name = MOCK_RIGETTI_GATE_MODEL_M_3_QPU["deviceName"]
        mock_m_3_device.arn = MOCK_RIGETTI_GATE_MODEL_M_3_QPU["deviceArn"]
        mock_m_3_device.provider = MOCK_RIGETTI_GATE_MODEL_M_3_QPU["providerName"]
        mock_m_3_device.status = MOCK_RIGETTI_GATE_MODEL_M_3_QPU["deviceStatus"]
        mock_m_3_device.device_type = MOCK_RIGETTI_GATE_MODEL_M_3_QPU["deviceType"]
        mock_m_3_device.device_capabilities = MOCK_RIGETTI_GATE_MODEL_M_3_QPU["deviceCapabilities"]
        mock_m_3_device_properties = MOCK_RIGETTI_M_3_QPU_CAPABILITIES
        mock_m_3_device_properties.service = Mock()
        mock_m_3_device_properties.service.updatedAt = "2023-06-02T17:00:00+00:00"
        mock_m_3_device.properties = mock_m_3_device_properties
        mock_m_3_device.gate_calibrations = None
        mock_m_3_device.type = "QPU"

        g = from_dict_of_lists(
            MOCK_RIGETTI_M_3_QPU_CAPABILITIES.paradigm.connectivity.connectivityGraph,
            create_using=DiGraph(),
        )
        mock_m_3_device.topology_graph = relabel_nodes(g, {n: int(n) for n in g.nodes})

        mock_get_devices.return_value = [mock_m_3_device]

        provider = BraketProvider()
        backend = provider.get_backend("Aspen-M-3")

        circ = qiskit_circuit.QuantumCircuit(4)
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.cx(2, 3)

        self.assertEqual(
            to_braket(circ, qubit_labels=backend.qubit_labels, target=backend.target).qubits,
            {0, 1, 2, 7},
        )

    @patch("qiskit_braket_provider.providers.braket_backend.BraketAwsBackend.run")
    @patch("qiskit_braket_provider.providers.braket_job.AmazonBraketTask.queue_position")
    @patch("qiskit_braket_provider.providers.braket_provider.AwsDevice")
    def test_queue_position_for_quantum_tasks(
        self, mocked_device: MagicMock, mock_queue_position: MagicMock, mock_run: MagicMock
    ) -> None:
        """Tests queue position for quantum tasks."""

        mock_return_value = QuantumTaskQueueInfo(
            queue_type=QueueType.NORMAL, queue_position=">2000", message=None
        )
        mock_task = Mock()
        mock_task.queue_position = mock_queue_position
        mock_queue_position.return_value = mock_return_value
        mock_run.return_value = mock_task

        mocked_device.properties = MOCK_RIGETTI_M_3_QPU_CAPABILITIES
        mocked_device.type = "QPU"
        backend = BraketAwsBackend(device=mocked_device)
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)

        qpu_task = backend.run(circuit, shots=1)
        result = qpu_task.queue_position()

        mock_queue_position.assert_called_once()
        assert isinstance(result, QuantumTaskQueueInfo)
        self.assertEqual(result, mock_return_value)

    def test_kraus_target_simulator(self) -> None:
        """test Kraus target works for multi-qubit kraus and we find multi-qubit Kraus operators"""

        k1 = [np.diag([0, 1]), np.diag([1, 0])]
        k2 = [np.diag([1, 1, 0, 0]), np.diag([0, 0, 1, 1])]

        qc = Circuit()
        qc.h(0)
        qc.h(1)
        qc.kraus([1], k1)
        qc.kraus([0, 1], k2)
        qc.kraus([2], k1)

        qd = BraketLocalBackend("braket_dm")
        c = transpile(to_qiskit(qc), backend=qd)
        assert c.count_ops()["kraus"] == 3
        nq = [ins.operation.num_qubits for ins in c.data if ins.operation.name == "kraus"]
        assert nq == [1, 2, 1]
