"""Braket instructions."""

from __future__ import annotations

from qiskit.circuit import ParameterExpression, QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library import CSXGate, XXPlusYYGate


## IQM Experimental capabilities
class MeasureFF(Instruction):
    """Measurement for Feed Forward control.

    Performs a measurement and stores the result in a classical feedback register
    for later use in conditional operations.

    Args:
        feedback_key (int): The integer feedback key that points to a measurement result.
    """

    def __init__(self, feedback_key: int) -> None:
        super().__init__("MeasureFF", 1, 0, params=[feedback_key])
        self.feedback_key = feedback_key

    broadcast_arguments = Gate.broadcast_arguments

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MeasureFF) and self.feedback_key == other.feedback_key

    def __hash__(self) -> int:
        return hash((self.name, self.feedback_key))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(feedback_key={self.feedback_key})"


class CCPRx(Instruction):
    """Classically controlled Phased Rx gate.

    A rotation around the X-axis with a phase factor, where the rotation depends
    on the value of a classical feedback.

    Args:
        angle_1 (float): The first angle of the gate in radians or
            expression representation.
        angle_2 (float): The second angle of the gate in radians or
            expression representation.
        feedback_key (int): The integer feedback key that points to a measurement result.
    """

    def __init__(self, angle_1: float, angle_2: float, feedback_key: int) -> None:
        super().__init__("CCPRx", 1, 0, params=[angle_1, angle_2, feedback_key])
        self.angle_1 = angle_1
        self.angle_2 = angle_2
        self.feedback_key = feedback_key

    broadcast_arguments = Gate.broadcast_arguments

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CCPRx)
            and self.feedback_key == other.feedback_key
            and self.angle_1 == other.angle_1
            and self.angle_2 == other.angle_2
        )

    def __hash__(self) -> int:
        return hash((self.name, self.angle_1, self.angle_2, self.feedback_key))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({self.angle_1}, {self.angle_2}, feedback_key={self.feedback_key})"
        )


class XY(Gate):
    """Braket xy gate. Equivalent to XXPlusYYGate(-theta, 0)."""

    def __init__(self, theta: float | ParameterExpression = 0.0, label: str | None = None) -> None:
        super().__init__("xy", 2, [theta], label=label)

    def _define(self) -> None:
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc.append(XXPlusYYGate(-self.params[0], 0), [q[0], q[1]])
        self.definition = qc


class _CPhaseShift(Gate):
    """Abstract base for cphaseshift{00, 01, 10}.

    Each subclass phases one basis state |AB> by e^{i*theta}. Since Qiskit's
    CPhaseGate phases only |11>, _define wraps it in X gates on the qubits
    listed in _CONJUGATE_X_QUBITS, which swap |AB> with |11> before and
    after the CPhase:

        cphaseshift00 -> X on both qubits  (|00> <-> |11>)
        cphaseshift01 -> X on qubit 0      (|01> <-> |11>)
        cphaseshift10 -> X on qubit 1      (|10> <-> |11>)
    """

    _BRAKET_NAME: str = ""
    _CONJUGATE_X_QUBITS: tuple[int, ...] = ()

    def __init__(self, theta: float | ParameterExpression = 0.0, label: str | None = None) -> None:
        if not self._BRAKET_NAME:
            raise TypeError(f"{type(self).__name__} is an abstract base class.")
        super().__init__(self._BRAKET_NAME, 2, [theta], label=label)

    def _define(self) -> None:
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        for i in self._CONJUGATE_X_QUBITS:
            qc.x(q[i])
        qc.cp(self.params[0], q[0], q[1])
        for i in self._CONJUGATE_X_QUBITS:
            qc.x(q[i])
        self.definition = qc


class CPhaseShift00(_CPhaseShift):
    _BRAKET_NAME = "cphaseshift00"
    _CONJUGATE_X_QUBITS = (0, 1)


class CPhaseShift01(_CPhaseShift):
    _BRAKET_NAME = "cphaseshift01"
    _CONJUGATE_X_QUBITS = (0,)


class CPhaseShift10(_CPhaseShift):
    _BRAKET_NAME = "cphaseshift10"
    _CONJUGATE_X_QUBITS = (1,)


class PSwap(Gate):
    """Braket pswap: SWAP followed by phase e^{i*theta} on |01> and |10>."""

    def __init__(self, theta: float | ParameterExpression = 0.0, label: str | None = None) -> None:
        super().__init__("pswap", 2, [theta], label=label)

    def _define(self) -> None:
        # pswap(theta) = SWAP @ diag(1, e^{i*theta}, e^{i*theta}, 1)
        #              = e^{i*theta/2} * SWAP @ RZZ(theta)
        theta = self.params[0]
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name, global_phase=theta / 2)
        qc.rzz(theta, q[0], q[1])
        qc.swap(q[0], q[1])
        self.definition = qc


class CV(Gate):
    """Braket cv (controlled sqrt-X). Same matrix as CSXGate."""

    def __init__(self, label: str | None = None) -> None:
        super().__init__("cv", 2, [], label=label)

    def _define(self) -> None:
        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q, name=self.name)
        qc.append(CSXGate(), [q[0], q[1]])
        self.definition = qc
