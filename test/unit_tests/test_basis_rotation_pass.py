"""Tests for AddBasisRotationGates TransformationPass."""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager

from braket.ir.jaqcd import Expectation, Probability, Sample, StateVector, Variance
from qiskit_braket_provider.providers.passes import AddBasisRotationGates


def _circuit_with_result_pragmas(num_qubits, pragmas):
    """Helper to create a circuit with result pragma metadata."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    if num_qubits > 1:
        qc.cx(0, 1)
    qc.metadata = {"braket_result_pragmas": pragmas}
    return qc


def test_z_observable_no_rotation():
    """Z observable requires no basis rotation gates."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result expectation z(q[0])",
            "parsed": Expectation(observable=["z"], targets=[0]),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert "h" in ops
    assert "cx" in ops
    assert "measure" in ops
    rotation_gates = [op for op in ops if op not in ("h", "cx", "measure", "barrier")]
    assert rotation_gates == []


def test_x_observable_adds_h_gate():
    """X observable requires an H gate for basis rotation."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result expectation x(q[0])",
            "parsed": Expectation(observable=["x"], targets=[0]),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert ops.count("h") == 2
    assert "measure" in ops


def test_y_observable_adds_z_s_h_gates():
    """Y observable requires Z, S, H gates for basis rotation."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result expectation y(q[0])",
            "parsed": Expectation(observable=["y"], targets=[0]),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert "z" in ops
    assert "s" in ops
    assert "measure" in ops


def test_h_observable_adds_ry_gate():
    """H observable requires Ry(-pi/4) for basis rotation."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result expectation h(q[0])",
            "parsed": Expectation(observable=["h"], targets=[0]),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert "ry" in ops
    assert "measure" in ops
    ry_instrs = [instr for instr in result.data if instr.operation.name == "ry"]
    assert len(ry_instrs) == 1
    assert math.isclose(ry_instrs[0].operation.params[0], -math.pi / 4)


def test_i_observable_no_rotation():
    """Identity observable requires no basis rotation gates."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result expectation i(q[0])",
            "parsed": Expectation(observable=["i"], targets=[0]),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert "measure" in ops
    rotation_gates = [op for op in ops if op not in ("h", "cx", "measure", "barrier")]
    assert rotation_gates == []


def test_tensor_product_observable():
    """Tensor product observable applies rotations per qubit."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result expectation x(q[0]) @ y(q[1])",
            "parsed": Expectation(observable=["x", "y"], targets=[0, 1]),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert "z" in ops
    assert "s" in ops
    assert ops.count("measure") == 2


def test_probability_result_type_no_rotation():
    """Probability result type is Z-basis, no rotation needed."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result probability q[0], q[1]",
            "parsed": Probability(targets=[0, 1]),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert "measure" in ops
    rotation_gates = [op for op in ops if op not in ("h", "cx", "measure", "barrier")]
    assert rotation_gates == []


def test_sample_result_type_with_observable():
    """Sample with an observable applies rotation gates."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result sample x(q[0])",
            "parsed": Sample(observable=["x"], targets=[0]),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert ops.count("h") == 2


def test_variance_result_type_with_observable():
    """Variance with an observable applies rotation gates."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result variance y(q[1])",
            "parsed": Variance(observable=["y"], targets=[1]),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert "z" in ops
    assert "s" in ops
    assert "measure" in ops


def test_no_result_pragmas_no_changes():
    """Circuit without result pragmas is unchanged."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert ops == ["h", "cx"]


def test_no_metadata_no_changes():
    """Circuit with no metadata is unchanged."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert ops == ["h", "cx"]


def test_measurements_added_for_all_targeted_qubits():
    """Measurements should be added for all qubits targeted by result types."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result expectation z(q[0]) @ z(q[1])",
            "parsed": Expectation(observable=["z", "z"], targets=[0, 1]),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    measure_ops = [instr for instr in result.data if instr.operation.name == "measure"]
    assert len(measure_ops) == 2


def test_state_vector_no_rotation_no_measurement():
    """StateVector result type should not add rotations or measurements."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result state_vector",
            "parsed": StateVector(),
        }
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert "measure" not in ops
    assert ops == ["h", "cx"]


def test_multiple_result_types_combined_rotations():
    """Multiple result types should combine their rotation requirements."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result expectation x(q[0])",
            "parsed": Expectation(observable=["x"], targets=[0]),
        },
        {
            "raw_pragma": "#pragma braket result variance y(q[1])",
            "parsed": Variance(observable=["y"], targets=[1]),
        },
    ]
    qc = _circuit_with_result_pragmas(2, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    assert "z" in ops
    assert "s" in ops
    assert ops.count("measure") == 2


def test_observable_targets_none_applies_to_all_qubits():
    """When targets is None, rotations should apply to all qubits."""
    pragmas = [
        {
            "raw_pragma": "#pragma braket result expectation x all",
            "parsed": Expectation(observable=["x"], targets=None),
        }
    ]
    qc = _circuit_with_result_pragmas(3, pragmas)

    pm = PassManager([AddBasisRotationGates()])
    result = pm.run(qc)

    ops = [instr.operation.name for instr in result.data]
    # x on all qubits -> H on each of 3 qubits
    assert ops.count("h") == 1 + 3  # original h + 3 rotations
    assert ops.count("measure") == 3
