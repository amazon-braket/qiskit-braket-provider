"""Tests for BraketSampler."""

from unittest import TestCase

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.primitives import BackendSamplerV2
from qiskit.primitives.containers import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from braket.circuits import Circuit
from braket.program_sets import CircuitBinding
from qiskit_braket_provider.providers import BraketLocalBackend
from qiskit_braket_provider.providers.braket_sampler import BraketSampler


class TestBraketSampler(TestCase):
    """Tests for BraketSampler."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        backend = BraketLocalBackend()
        self.sampler = BraketSampler(backend)
        self.sampler_backend = BackendSamplerV2(backend=backend)

    def assert_correct_results(self, actual: BitArray, expected: BitArray) -> None:
        """Compares the results from BraketSampler and BackendSamplerV2"""
        counts = actual.get_int_counts()
        shots = actual.num_shots

        counts_backend = expected.get_int_counts()
        shots_backend = expected.num_shots

        self.assertEqual(counts.keys(), counts_backend.keys())
        for k, v in counts.items():
            self.assertTrue(
                np.isclose(v / shots, counts_backend[k] / shots_backend, rtol=0.3, atol=0.2)
            )

    def test_program_sets_unsupported(self) -> None:
        """Tests that initialization raises a ValueError if program sets aren't supported"""
        backend = BraketLocalBackend()
        backend._supports_program_sets = False
        with self.assertRaises(ValueError):
            BraketSampler(backend)

    def test_different_shots_raises_error(self) -> None:
        """Test that pubs with different shots raise an error."""
        theta = Parameter("θ")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        with self.assertRaises(ValueError) as context:
            self.sampler.run([(qc, [0, 1, 2, 3], 100), (qc, [0, 1, 2, 3], 200)])

        self.assertIn("same shots", str(context.exception))

    def test_run_local_multiple_registers(self) -> None:
        """Tests that correct results are returned for circuits with multiple registers"""
        circuit = QuantumCircuit(
            QuantumRegister(3, "qreg_a"),
            QuantumRegister(9, "qreg_b"),
            ClassicalRegister(10, "creg_a"),
            ClassicalRegister(2, "creg_b"),
        )
        circuit.h(0)
        for i in range(11):
            circuit.cx(i, i + 1)
        circuit.ry(Parameter("θ"), 0)
        circuit.measure_all(add_bits=False)

        num_steps = 6
        pub = (
            circuit,
            np.array(  # shape (3, 6)
                [
                    np.linspace(0, 2 * np.pi, num_steps),
                    np.linspace(0, np.pi, num_steps),
                    np.linspace(np.pi, 2 * np.pi, num_steps),
                ]
            ),
        )
        coerced = SamplerPub.coerce(pub)

        task = self.sampler.run([pub])
        program_set = task.program_set
        self.assertEqual(len(program_set), 1)
        self.assertEqual(len(program_set[0]), coerced.size)
        data = task.result()[0].data
        data_backend = self.sampler_backend.run([pub]).result()[0].data
        for reg, reg_backend in [
            (data.creg_a, data_backend.creg_a),
            (data.creg_b, data_backend.creg_b),
        ]:
            for index in np.ndindex(coerced.shape):
                self.assert_correct_results(reg[index], reg_backend[index])

    def test_run_local_shapeless_parameters(self) -> None:
        """Tests that correct results are returned for circuits with no or shapeless parameters"""
        qc1 = QuantumCircuit(2)
        qc1.id(0)
        qc1.x(0)
        qc1.measure_all()
        qc2 = QuantumCircuit(2)
        qc2.h(0)
        qc2.cx(0, 1)
        qc2.ry(Parameter("θ"), 0)
        qc2.measure_all()
        pubs = [(qc1,), (qc2, np.pi / 3)]
        task = self.sampler.run(pubs)
        program_set = task.program_set
        self.assertEqual(len(program_set), 2)
        self.assertTrue(isinstance(program_set[0], Circuit))
        self.assertTrue(isinstance(program_set[1], CircuitBinding))
        self.assertEqual(program_set.total_executables, 2)
        for actual, expected in zip(
            task.result(), self.sampler_backend.run(pubs).result(), strict=True
        ):
            self.assert_correct_results(actual.data.meas, expected.data.meas)
