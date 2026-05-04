"""Braket instructions."""

from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction


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
