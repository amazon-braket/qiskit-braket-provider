"""Tests for Braket instructions."""

import unittest

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import CircuitInstruction, Parameter, QuantumRegister, Qubit
from qiskit.quantum_info import Operator

from braket.circuits import Circuit
from braket.circuits import gates as braket_gates
from braket.experimental_capabilities import EnableExperimentalCapability
from qiskit_braket_provider import to_braket, to_qiskit
from qiskit_braket_provider.providers.adapter import _default_target
from qiskit_braket_provider.providers.braket_instructions import (
    CV,
    XY,
    CCPRx,
    CPhaseShift00,
    CPhaseShift01,
    CPhaseShift10,
    MeasureFF,
    PSwap,
    _CPhaseShift,
)


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


@pytest.mark.parametrize(
    ("qiskit_cls", "name", "params"),
    [
        (XY, "xy", (0.5,)),
        (CPhaseShift00, "cphaseshift00", (0.5,)),
        (CPhaseShift01, "cphaseshift01", (0.5,)),
        (CPhaseShift10, "cphaseshift10", (0.5,)),
        (PSwap, "pswap", (0.5,)),
        (CV, "cv", ()),
    ],
    ids=["xy", "cphaseshift00", "cphaseshift01", "cphaseshift10", "pswap", "cv"],
)
def test_openqasm_round_trip_preserves_name_and_unitary(
    qiskit_cls: type, name: str, params: tuple[float, ...]
) -> None:
    args = f"({params[0]})" if params else ""
    qasm = f"OPENQASM 3.0;\n{name}{args} $0, $1;\n"
    qc = to_qiskit(qasm)
    assert isinstance(qc.data[0].operation, qiskit_cls)

    braket_circuit = to_braket(qc)
    braket_cls = getattr(braket_gates, qiskit_cls.__name__)
    assert isinstance(braket_circuit.instructions[0].operator, braket_cls)

    ref = Circuit.from_ir(qasm).to_unitary()
    assert np.allclose(braket_circuit.to_unitary(), ref)
    assert np.allclose(Operator(qc).reverse_qargs().data, ref)


@pytest.mark.parametrize(
    ("qiskit_cls", "name", "params"),
    [
        (XY, "xy", (0.5,)),
        (CPhaseShift00, "cphaseshift00", (0.5,)),
        (CPhaseShift01, "cphaseshift01", (0.5,)),
        (CPhaseShift10, "cphaseshift10", (0.5,)),
        (PSwap, "pswap", (0.5,)),
        (CV, "cv", ()),
    ],
    ids=["xy", "cphaseshift00", "cphaseshift01", "cphaseshift10", "pswap", "cv"],
)
def test_transpile_to_basic_basis_preserves_unitary(
    qiskit_cls: type, name: str, params: tuple[float, ...]
) -> None:
    qc = QuantumCircuit(2)
    qc.append(qiskit_cls(*params), [0, 1])
    tqc = transpile(qc, basis_gates=["u", "cx"], optimization_level=1)
    assert not any(inst.operation.name == name for inst in tqc.data)
    assert np.allclose(Operator(qc).to_matrix(), Operator(tqc).to_matrix())


def test_direct_instantiation_of_abstract_base_raises() -> None:
    with pytest.raises(TypeError):
        _CPhaseShift(0.5)


def test_to_qiskit_supports_iqm_classical_control() -> None:
    """Regression test for to_qiskit on cc_prx and measure_ff."""
    qasm = "OPENQASM 3.0;\nqubit[2] q;\nmeasure_ff(0) q[0];\ncc_prx(0.5, 0.7, 0) q[1];\n"
    with EnableExperimentalCapability():
        qc = to_qiskit(Circuit.from_ir(qasm), add_measurements=False)
    assert isinstance(qc.data[0].operation, MeasureFF)
    assert isinstance(qc.data[1].operation, CCPRx)


if __name__ == "__main__":
    unittest.main()
