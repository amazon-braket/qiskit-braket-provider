"""Tests for Braket instructions."""

import unittest

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Parameter, QuantumRegister, Qubit

from braket.experimental_capabilities import EnableExperimentalCapability
from qiskit_braket_provider import to_braket
from qiskit_braket_provider.providers.adapter import _default_target
from qiskit_braket_provider.providers.braket_instructions import CCPRx, MeasureFF


class TestIqmExperimentalCapabilities(unittest.TestCase):
    """Tests for Braket instructions."""

    def test_measureff_initialization(self):
        """Test MeasureFF initialization with valid parameters"""
        feedback_key = 1
        measure = MeasureFF(feedback_key)

        self.assertEqual(measure.name, "MeasureFF")
        self.assertEqual(measure.num_qubits, 1)
        self.assertEqual(measure.num_clbits, 0)
        self.assertEqual(measure.params, [feedback_key])

    def test_measureff_equality(self):
        """Test MeasureFF equality comparison"""
        measure1 = MeasureFF(1)
        measure2 = MeasureFF(1)
        measure3 = MeasureFF(2)

        self.assertEqual(measure1, measure2)
        self.assertNotEqual(measure1, measure3)
        self.assertNotEqual(measure1, "not_a_measure")
        self.assertEqual(hash(measure1), hash(measure2))
        self.assertNotEqual(hash(measure1), hash(measure3))

    def test_measureff_repr(self):
        """Test MeasureFF string representation"""
        measure = MeasureFF(1)
        expected_repr = "MeasureFF(feedback_key=1)"
        self.assertEqual(repr(measure), expected_repr)

    def test_ccprx_initialization(self):
        """Test CCPRx initialization with valid parameters"""
        angle1 = 0.5
        angle2 = 0.7
        feedback_key = 1
        ccprx = CCPRx(angle1, angle2, feedback_key)

        self.assertEqual(ccprx.name, "CCPRx")
        self.assertEqual(ccprx.num_qubits, 1)
        self.assertEqual(ccprx.num_clbits, 0)
        self.assertEqual(ccprx.params, [angle1, angle2, feedback_key])

    def test_ccprx_equality(self):
        """Test CCPRx equality comparison"""
        ccprx1 = CCPRx(0.5, 0.7, 1)
        ccprx2 = CCPRx(0.5, 0.7, 1)
        ccprx3 = CCPRx(0.5, 0.7, 2)
        ccprx4 = CCPRx(0.6, 0.7, 1)

        self.assertEqual(ccprx1, ccprx2)
        self.assertNotEqual(ccprx1, ccprx3)
        self.assertNotEqual(ccprx1, ccprx4)
        self.assertNotEqual(ccprx1, "not_a_ccprx")
        self.assertEqual(hash(ccprx1), hash(ccprx2))
        self.assertNotEqual(hash(ccprx1), hash(ccprx3))

    def test_ccprx_repr(self):
        """Test CCPRx string representation"""
        ccprx = CCPRx(0.5, 0.7, 1)
        expected_repr = "CCPRx(0.5, 0.7, feedback_key=1)"
        self.assertEqual(repr(ccprx), expected_repr)

    def test_circuit_with_measureff_ccprx(self):
        """Test circuit with MeasureFF instruction"""
        circuit = QuantumCircuit(1, 1)
        circuit.append(MeasureFF(feedback_key=0), qargs=[0])
        circuit.append(CCPRx(0.5, 0.7, feedback_key=0), qargs=[0])

        assert circuit.data[0] == CircuitInstruction(
            MeasureFF(0), qubits=(Qubit(QuantumRegister(1, "q"), 0),)
        )
        assert circuit.data[1] == CircuitInstruction(
            CCPRx(0.5, 0.7, 0), qubits=(Qubit(QuantumRegister(1, "q"), 0),)
        )

        target = _default_target([circuit])
        target.add_instruction(
            CCPRx(Parameter("angle_1"), Parameter("angle_2"), Parameter("feedback_key"))
        )
        target.add_instruction(MeasureFF(Parameter("feedback_key")))

        with EnableExperimentalCapability():
            braket_circuit = to_braket(circuit, target=target)

        assert braket_circuit.instructions[0].operator.name == "MeasureFF"
        assert braket_circuit.instructions[0].operator.parameters == [0]
        assert braket_circuit.instructions[0].target == [0]
        assert braket_circuit.instructions[1].operator.name == "CCPRx"
        assert braket_circuit.instructions[1].operator.parameters == [0.5, 0.7, 0]
        assert braket_circuit.instructions[1].target == [0]


if __name__ == "__main__":
    unittest.main()
