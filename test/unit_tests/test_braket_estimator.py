"""Tests for BraketEstimator."""

from collections.abc import Iterable
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimatorV2, BasePrimitiveJob
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub, EstimatorPubLike
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp

from braket.program_sets import ProgramSet
from qiskit_braket_provider.providers import BraketLocalBackend
from qiskit_braket_provider.providers.braket_estimator import BraketEstimator
from qiskit_braket_provider.providers.braket_primitive_task import BraketPrimitiveTask


class TestBraketEstimator(TestCase):
    """Tests for BraketEstimator."""

    def setUp(self):
        """Set up test fixtures."""
        self.backend = BraketLocalBackend()
        self.estimator = BraketEstimator(self.backend)
        self.estimator_backend = BackendEstimatorV2(backend=self.backend)

    def assert_correct_results(self, task: BraketPrimitiveTask, pubs: Iterable[EstimatorPubLike]):
        """Compares the results from BraketEstimator and BackendEstimatorV2"""
        for actual, expected in zip(
            task.result(), self.estimator_backend.run(pubs).result(), strict=True
        ):
            self.assertTrue(np.allclose(actual.data.evs, expected.data.evs, rtol=0.3, atol=0.2))

    def test_program_sets_unsupported(self):
        """Tests that initialization raises a ValueError if program sets aren't supported"""
        backend = BraketLocalBackend()
        backend._supports_program_sets = False
        with self.assertRaises(ValueError):
            BraketEstimator(backend)

    def test_simple_pub(self):
        """Test a simple pub with no broadcasting."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        observable = SparsePauliOp(["ZZ"])
        pub = (qc, observable)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            self.assertIsInstance(call_args[0][0], ProgramSet)
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_parameterized_circuit(self):
        """Test with a parameterized circuit."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        observable = SparsePauliOp(["Z"])
        param_values = np.array([[0.0], [np.pi / 4], [np.pi / 2]])
        pub = (qc, observable, param_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_multiple_observables(self):
        """Test with multiple observables."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        observables = [SparsePauliOp(["ZZ"]), SparsePauliOp(["XX"])]
        pub = (qc, observables)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            program_set = mock_run.call_args[0][0]
            self.assertIsInstance(program_set, ProgramSet)

    def test_multiple_pubs(self):
        """Test running multiple pubs."""
        qc1 = QuantumCircuit(1)
        qc1.h(0)

        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)

        obs1 = SparsePauliOp(["Z"])
        obs2 = SparsePauliOp(["ZZ"])

        pub1 = (qc1, obs1)
        pub2 = (qc2, obs2)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub1, pub2], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_default_precision(self):
        """Test that default precision is used when not specified."""
        qc = QuantumCircuit(1)
        qc.h(0)
        observable = SparsePauliOp(["Z"])
        pub = (qc, observable)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub])

            self.assertIsInstance(job, BasePrimitiveJob)

    def test_custom_precision(self):
        """Test using custom precision."""
        qc = QuantumCircuit(1)
        qc.h(0)
        observable = SparsePauliOp(["Z"])
        pub = (qc, observable)

        custom_precision = 0.05

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=custom_precision)

            self.assertIsInstance(job, BasePrimitiveJob)

    def test_complex_broadcasting(self):
        """Test with complex broadcasting shapes (2, 3, 6)."""
        theta = Parameter("θ")
        phi = Parameter("φ")
        qc = QuantumCircuit(2)
        qc.ry(theta, 0)
        qc.rx(phi, 1)
        qc.cx(0, 1)

        # Parameter values with shape (3, 6)
        rng = np.random.default_rng(42)
        param_data = {
            "θ": rng.uniform(0, 2 * np.pi, size=(3, 6)),
            "φ": rng.uniform(0, 2 * np.pi, size=(3, 6)),
        }
        parameter_values = BindingsArray(param_data)

        # Observables with shape (2, 3, 1)
        observables = [
            [[SparsePauliOp(["ZZ"])], [SparsePauliOp(["XX"])], [SparsePauliOp(["YY"])]],
            [[SparsePauliOp(["ZI"])], [SparsePauliOp(["IZ"])], [SparsePauliOp(["XI"])]],
        ]

        pub = (qc, observables, parameter_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            program_set = mock_run.call_args[0][0]
            self.assertIsInstance(program_set, ProgramSet)

            self.assertIsInstance(job, BasePrimitiveJob)

    def test_broadcasting_with_scalar_observable(self):
        """Test broadcasting with scalar observable and array parameters."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        param_values = np.linspace(0, np.pi, 5)
        observable = SparsePauliOp(["Z"])
        pub = (qc, observable, param_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_broadcasting_with_array_observables(self):
        """Test broadcasting with array observables and scalar parameters."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        observables = [
            SparsePauliOp(["ZZ"]),
            SparsePauliOp(["XX"]),
            SparsePauliOp(["YY"]),
            SparsePauliOp(["ZI"]),
        ]

        pub = (qc, observables)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            job = self.estimator.run([pub], precision=0.01)

            mock_run.assert_called_once()
            self.assertIsInstance(job, BasePrimitiveJob)

    def test_different_precisions_raises_error(self):
        """Test that pubs with different precisions raise an error."""
        qc = QuantumCircuit(1)
        qc.h(0)
        observable = SparsePauliOp(["Z"])

        # Create pubs with different precisions
        obs_array = ObservablesArray([observable])
        pub1 = EstimatorPub(circuit=qc, observables=obs_array, precision=0.01)
        pub2 = EstimatorPub(circuit=qc, observables=obs_array, precision=0.02)

        with self.assertRaises(ValueError) as context:
            self.estimator.run([pub1, pub2])

        self.assertIn("same precision", str(context.exception))

    def test_non_broadcastable_shapes_raises_error(self):
        """Test that non-broadcastable shapes raise an error."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)

        # Create observables with shape (3,)
        observables = [SparsePauliOp(["Z"]), SparsePauliOp(["X"]), SparsePauliOp(["Y"])]

        # Create parameter values with shape (2, 1) - not broadcastable with (3,)
        param_values = np.array([[0.0], [np.pi / 4]])

        pub = (qc, observables, param_values)

        with patch.object(self.backend._device, "run") as mock_run:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_run.return_value = mock_task

            with self.assertRaises(ValueError) as context:
                self.estimator.run([pub], precision=0.01)

            self.assertIn("not broadcastable", str(context.exception))

    def test_run_local_single_observable_or_parameter(self):
        """Tests that correct results are returned when there is only one observable or parameter"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("θ"), 0)
        observables = [SparsePauliOp("ZX"), SparsePauliOp("XZ")]
        parameters = [np.pi / 3, np.pi / 6]
        pubs = [
            (circuit, observables[0], parameters),
            (circuit, observables, parameters[0]),
            (circuit, observables[1], parameters[1]),
        ]
        task = self.estimator.run(pubs, abelian_grouping=False)
        program_set = task.program_set
        self.assertEqual(len(program_set), 3)
        self.assertEqual(len(program_set[0]), 2)
        self.assertEqual(len(program_set[1]), 2)
        self.assertEqual(len(program_set[2]), 1)
        self.assert_correct_results(task, pubs)

    def test_run_local_no_parameters(self):
        """Tests that correct results are returned for circuits with no parameters"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(np.pi / 3, 0)
        observables = [SparsePauliOp("ZX"), SparsePauliOp("XZ")]
        pubs = [(circuit, observables), (circuit, observables[0])]
        task = self.estimator.run(pubs, abelian_grouping=False)
        program_set = task.program_set
        self.assertEqual(len(program_set), 2)
        self.assertEqual(len(program_set[0]), 2)
        self.assertEqual(len(program_set[1]), 1)
        self.assert_correct_results(task, pubs)

    def test_run_local_pauli_sum(self):
        """Tests that correct results are returned when one observable is a Pauli sum"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("θ[0]"), 0)
        circuit.rz(Parameter("θ[1]"), 0)
        circuit.cx(0, 1)
        circuit.h(0)

        num_params = 20
        pub = (
            circuit,
            [
                [SparsePauliOp("ZZ")],
                [SparsePauliOp("ZX")],
                [SparsePauliOp("XZ")],
                [SparsePauliOp(["ZX", "XZ"], [0.3, 0.8])],
            ],
            np.vstack([
                np.linspace(-np.pi, np.pi, num_params),
                np.linspace(-4 * np.pi, 4 * np.pi, num_params),
            ]).T,
        )

        task = self.estimator.run([pub], abelian_grouping=False)
        program_set = task.program_set
        self.assertEqual(len(program_set), 2)
        self.assertEqual(len(program_set[0]), num_params * 2)
        self.assertEqual(len(program_set[1]), num_params * 3)
        self.assert_correct_results(task, [pub])

    def test_run_local_all_pauli_sums(self):
        """Tests that correct results are returned when all observables are Pauli sums"""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("theta"), 0)
        circuit.rz(Parameter("phi"), 0)
        circuit.cx(0, 1)
        circuit.h(0)

        num_params = 20
        pub = (
            circuit,
            [
                [SparsePauliOp(["XX", "IY"], [0.5, 0.5])],
                [SparsePauliOp(["YY", "ZI", "XY"], [0.5, 0.5, 0.1])],
            ],
            np.vstack([
                np.linspace(-np.pi, np.pi, num_params),
                np.linspace(-4 * np.pi, 4 * np.pi, num_params),
            ]).T,
        )

        task = self.estimator.run([pub], abelian_grouping=False)
        program_set = task.program_set
        self.assertEqual(len(program_set), 2)
        self.assertEqual(len(program_set[0]), num_params * 2)
        self.assertEqual(len(program_set[1]), num_params * 3)
        self.assert_correct_results(task, [pub])

    def test_run_local_broadcasting(self):
        """Tests that correct results are returned with broadcasted arrays"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("θ"), 0)
        circuit.h(2)

        num_steps = 6
        pub = (
            circuit,
            [
                [[SparsePauliOp(["IZZ"])], [SparsePauliOp(["IZX"])], [SparsePauliOp(["III"])]],
                [[SparsePauliOp(["IXZ"])], [SparsePauliOp(["IXX"])], [SparsePauliOp(["YYY"])]],
            ],
            np.array(  # shape (3, 6)
                [
                    np.linspace(0, 2 * np.pi, num_steps),
                    np.linspace(0, np.pi, num_steps),
                    np.linspace(np.pi, 2 * np.pi, num_steps),
                ]
            ),
        )

        task = self.estimator.run([pub], abelian_grouping=False)
        program_set = task.program_set
        self.assertEqual(len(program_set), 3)
        for entry in task.program_set:
            self.assertEqual(len(entry), num_steps * 2)
        self.assert_correct_results(task, [pub])

    def test_abelian_grouping_collapses_commuting_group(self):
        """A commuting group is measured in one covering executable, not one per term."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        obs = SparsePauliOp(["ZZ", "IZ", "ZI"], [1.0, 0.5, -0.3])
        pub = (circuit, obs)

        grouped = self.estimator.run([pub])
        ungrouped = self.estimator.run([pub], abelian_grouping=False)

        self.assertEqual(len(grouped.program_set[0]), 1)
        self.assertEqual(len(ungrouped.program_set[0]), 3)
        self.assert_correct_results(grouped, [pub])

    def test_abelian_grouping_reuses_shared_terms(self):
        """Two observables sharing a term reconstruct from one shared measurement."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        pub = (
            circuit,
            [
                SparsePauliOp(["ZI", "ZZ"], [0.25, 0.75]),
                SparsePauliOp("ZI"),
            ],
        )
        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 1)
        self.assert_correct_results(task, [pub])

    def test_abelian_grouping_parameterized(self):
        """Grouping reconstructs correctly across a parameter sweep."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("θ"), 0)
        obs = SparsePauliOp(["ZZ", "IZ", "ZI"], [1.0, 0.5, -0.3])
        num_params = 5
        pub = (circuit, obs, np.linspace(0, np.pi, num_params))

        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), num_params)
        self.assert_correct_results(task, [pub])

    def test_abelian_grouping_identity_with_active_terms(self):
        """An identity term alongside active terms is added with no extra executable."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        obs = SparsePauliOp(["II", "ZZ"], [0.5, 1.0])
        pub = (circuit, obs)
        task = self.estimator.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), 1)
        self.assert_correct_results(task, [pub])

    def test_abelian_grouping_pure_identity(self):
        """A pure-identity (constant) observable returns the constant without crashing."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        obs = SparsePauliOp(["II"], [2.0])
        pub = (circuit, obs)
        task = self.estimator.run([pub])
        self.assert_correct_results(task, [pub])

    def test_abelian_grouping_y_basis_group(self):
        """A Y-basis commuting group is measured in one covering executable."""
        circuit = QuantumCircuit(2)
        circuit.ry(0.7, 0)
        circuit.rx(1.1, 1)
        circuit.cx(0, 1)
        obs = SparsePauliOp(["YY", "YI", "IY"])
        pub = (circuit, obs)
        task = self.estimator.run([pub])
        self.assertEqual(len(task.program_set), 1)
        self.assertEqual(len(task.program_set[0]), 1)
        self.assert_correct_results(task, [pub])

    def test_abelian_grouping_mixed_basis_group(self):
        """Terms needing different bases on different qubits share one covering measurement."""
        circuit = QuantumCircuit(2)
        circuit.ry(0.7, 0)
        circuit.rx(1.1, 1)
        circuit.cx(0, 1)
        obs = SparsePauliOp(["ZI", "IX", "ZX"])
        pub = (circuit, obs)
        task = self.estimator.run([pub])
        self.assertEqual(len(task.program_set), 1)
        self.assertEqual(len(task.program_set[0]), 1)
        self.assert_correct_results(task, [pub])

    def test_abelian_grouping_multiple_groups(self):
        """An observable spanning two commuting groups uses one executable per group."""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(0.6, 2)
        circuit.cx(1, 2)
        obs = SparsePauliOp(
            ["ZZI", "IZZ", "XII", "IXI", "IIX"],
            [-1.0, -1.0, -0.5, -0.5, -0.5],
        )
        pub = (circuit, obs)
        task = self.estimator.run([pub])
        self.assertEqual(len(task.program_set), 2)
        self.assert_correct_results(task, [pub])

    def test_abelian_grouping_four_qubit_fewer_executions(self):
        """On four qubits, grouping reconstructs correctly with fewer total executions."""
        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(0.6, 2)
        circuit.cx(2, 3)
        circuit.cx(1, 2)
        obs = SparsePauliOp(
            ["ZZII", "IZZI", "IIZZ", "XIII", "IXII", "IIXI", "IIIX"],
            [-1.0, -1.0, -1.0, -0.5, -0.5, -0.5, -0.5],
        )
        pub = (circuit, obs)
        grouped = self.estimator.run([pub])
        ungrouped = self.estimator.run([pub], abelian_grouping=False)
        grouped_execs = sum(len(entry) for entry in grouped.program_set)
        ungrouped_execs = sum(len(entry) for entry in ungrouped.program_set)
        self.assertEqual(ungrouped_execs, 7)
        self.assertEqual(grouped_execs, 2)
        self.assert_correct_results(grouped, [pub])

    def test_abelian_grouping_broadcasts_over_2d_shape(self):
        """Grouping reconstructs every cell of a 2-D (observable x parameter) broadcast."""
        theta = Parameter("θ")
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(theta, 0)
        observables = ObservablesArray([
            [SparsePauliOp(["ZZ", "IZ", "ZI"], [1.0, 0.5, -0.3])],
            [SparsePauliOp(["XX", "XI", "IX"], [1.0, 0.5, 0.25])],
        ])
        parameter_values = BindingsArray({theta: np.array([0.0, np.pi / 4, np.pi / 2])})
        pub = (circuit, observables, parameter_values)
        self.assertEqual(EstimatorPub.coerce(pub).shape, (2, 3))
        task = self.estimator.run([pub])
        self.assert_correct_results(task, [pub])
