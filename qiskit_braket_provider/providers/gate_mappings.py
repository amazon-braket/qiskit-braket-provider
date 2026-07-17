"""Shared constants and gate mapping tables.

This module contains the gate mapping dictionaries and constants used across
adapter.py, target.py, qasm_context.py, and compilation.py.
"""

from collections.abc import Callable
from math import inf, pi

import qiskit.circuit.library as qiskit_gates
import qiskit.quantum_info as qiskit_qi
from qiskit.circuit import Instruction as QiskitInstruction
from qiskit.circuit import Parameter
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit_ionq import ionq_gates
from sympy import acos, asin, atan, cos, exp, log, sin, tan

from braket import experimental_capabilities as braket_expcaps
from braket.circuits import gates as braket_gates
from braket.circuits import noises as braket_noises
from braket.circuits import observables as braket_observables
from braket.ir.jaqcd import (
    Amplitude,
    DensityMatrix,
    Expectation,
    Probability,
    Sample,
    StateVector,
    Variance,
)
from qiskit_braket_provider.providers import braket_instructions

_BRAKET_TO_QISKIT_NAMES = {
    "u": "u",
    "phaseshift": "p",
    "cnot": "cx",
    "x": "x",
    "y": "y",
    "z": "z",
    "t": "t",
    "ti": "tdg",
    "s": "s",
    "si": "sdg",
    "v": "sx",
    "vi": "sxdg",
    "swap": "swap",
    "iswap": "iswap",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "xx": "rxx",
    "yy": "ryy",
    "zz": "rzz",
    "xy": "xy",
    "i": "id",
    "h": "h",
    "cy": "cy",
    "cz": "cz",
    "ccnot": "ccx",
    "cswap": "cswap",
    "cphaseshift": "cp",
    "cphaseshift00": "cphaseshift00",
    "cphaseshift01": "cphaseshift01",
    "cphaseshift10": "cphaseshift10",
    "pswap": "pswap",
    "cv": "cv",
    "ecr": "ecr",
    "prx": "r",
    "gpi": "gpi",
    "gpi2": "gpi2",
    "ms": "ms",
    "gphase": "global_phase",
    "unitary": "unitary",
    "kraus": "kraus",
}

_CONTROLLED_GATES_BY_QUBIT_COUNT = {
    1: {
        "ch": "h",
        "cs": "s",
        "csdg": "sdg",
        "csx": "sx",
        "crx": "rx",
        "cry": "ry",
        "crz": "rz",
        "ccz": "cz",
    },
    3: {"c3sx": "sx"},
    inf: {"mcx": "cx"},
}


_QISKIT_GATE_NAME_TO_BRAKET_GATE: dict[str, Callable] = {
    "u1": lambda lam: [braket_gates.U(0, 0, lam)],
    "u2": lambda phi, lam: [braket_gates.U(pi / 2, phi, lam)],
    "u3": lambda theta, phi, lam: [braket_gates.U(theta, phi, lam)],
    "u": lambda theta, phi, lam: [braket_gates.U(theta, phi, lam)],
    "p": lambda angle: [braket_gates.PhaseShift(angle)],
    "cp": lambda angle: [braket_gates.CPhaseShift(angle)],
    "cx": lambda: [braket_gates.CNot()],
    "x": lambda: [braket_gates.X()],
    "y": lambda: [braket_gates.Y()],
    "z": lambda: [braket_gates.Z()],
    "t": lambda: [braket_gates.T()],
    "tdg": lambda: [braket_gates.Ti()],
    "s": lambda: [braket_gates.S()],
    "sdg": lambda: [braket_gates.Si()],
    "sx": lambda: [braket_gates.V()],
    "sxdg": lambda: [braket_gates.Vi()],
    "swap": lambda: [braket_gates.Swap()],
    "rx": lambda angle: [braket_gates.Rx(angle)],
    "ry": lambda angle: [braket_gates.Ry(angle)],
    "rz": lambda angle: [braket_gates.Rz(angle)],
    "rzz": lambda angle: [braket_gates.ZZ(angle)],
    "id": lambda: [braket_gates.I()],
    "h": lambda: [braket_gates.H()],
    "cy": lambda: [braket_gates.CY()],
    "cz": lambda: [braket_gates.CZ()],
    "ccx": lambda: [braket_gates.CCNot()],
    "cswap": lambda: [braket_gates.CSwap()],
    "rxx": lambda angle: [braket_gates.XX(angle)],
    "ryy": lambda angle: [braket_gates.YY(angle)],
    "ecr": lambda: [braket_gates.ECR()],
    "iswap": lambda: [braket_gates.ISwap()],
    "xy": lambda angle: [braket_gates.XY(angle)],
    "cphaseshift00": lambda angle: [braket_gates.CPhaseShift00(angle)],
    "cphaseshift01": lambda angle: [braket_gates.CPhaseShift01(angle)],
    "cphaseshift10": lambda angle: [braket_gates.CPhaseShift10(angle)],
    "pswap": lambda angle: [braket_gates.PSwap(angle)],
    "cv": lambda: [braket_gates.CV()],
    "r": lambda angle_1, angle_2: [braket_gates.PRx(angle_1, angle_2)],
    # IonQ gates
    "gpi": lambda turns: [braket_gates.GPi(2 * pi * turns)],
    "gpi2": lambda turns: [braket_gates.GPi2(2 * pi * turns)],
    "ms": lambda turns_1, turns_2, turns_3: [
        braket_gates.MS(2 * pi * turns_1, 2 * pi * turns_2, 2 * pi * turns_3)
    ],
    "zz": lambda angle: [braket_gates.ZZ(2 * pi * angle)],
    # Global phase
    "global_phase": lambda phase: [braket_gates.GPhase(phase)],
    "unitary": lambda operators: [braket_gates.Unitary(operators[0])],
    "kraus": lambda operators: [braket_noises.Kraus(operators)],
    "CCPRx": lambda angle_1, angle_2, feedback_key: [
        braket_expcaps.iqm.classical_control.CCPRx(angle_1, angle_2, feedback_key)
    ],
    "MeasureFF": lambda feedback_key: [
        braket_expcaps.iqm.classical_control.MeasureFF(feedback_key)
    ],
}

_QISKIT_CONTROLLED_GATE_NAMES_TO_BRAKET_GATES: dict[str, Callable] = {
    controlled_gate: _QISKIT_GATE_NAME_TO_BRAKET_GATE[base_gate]
    for gate_map in _CONTROLLED_GATES_BY_QUBIT_COUNT.values()
    for controlled_gate, base_gate in gate_map.items()
}

_STANDARD_GATE_NAME_MAPPING = get_standard_gate_name_mapping()

_BRAKET_GATE_NAME_TO_QISKIT_GATE: dict[str, QiskitInstruction | None] = {
    "u": qiskit_gates.UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
    "u1": qiskit_gates.U1Gate(Parameter("theta")),
    "u2": qiskit_gates.U2Gate(Parameter("theta"), Parameter("lam")),
    "u3": qiskit_gates.U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
    "h": qiskit_gates.HGate(),
    "ccnot": qiskit_gates.CCXGate(),
    "cnot": qiskit_gates.CXGate(),
    "cphaseshift": qiskit_gates.CPhaseGate(Parameter("theta")),
    "cswap": qiskit_gates.CSwapGate(),
    "cy": qiskit_gates.CYGate(),
    "cz": qiskit_gates.CZGate(),
    "i": qiskit_gates.IGate(),
    "phaseshift": qiskit_gates.PhaseGate(Parameter("theta")),
    "rx": qiskit_gates.RXGate(Parameter("theta")),
    "ry": qiskit_gates.RYGate(Parameter("theta")),
    "rz": qiskit_gates.RZGate(Parameter("phi")),
    "s": qiskit_gates.SGate(),
    "si": qiskit_gates.SdgGate(),
    "swap": qiskit_gates.SwapGate(),
    "t": qiskit_gates.TGate(),
    "ti": qiskit_gates.TdgGate(),
    "v": qiskit_gates.SXGate(),
    "vi": qiskit_gates.SXdgGate(),
    "x": qiskit_gates.XGate(),
    "xx": qiskit_gates.RXXGate(Parameter("theta")),
    "y": qiskit_gates.YGate(),
    "yy": qiskit_gates.RYYGate(Parameter("theta")),
    "xy": braket_instructions.XY(Parameter("theta")),
    "z": qiskit_gates.ZGate(),
    "zz": qiskit_gates.RZZGate(Parameter("theta")),
    "cphaseshift00": braket_instructions.CPhaseShift00(Parameter("theta")),
    "cphaseshift01": braket_instructions.CPhaseShift01(Parameter("theta")),
    "cphaseshift10": braket_instructions.CPhaseShift10(Parameter("theta")),
    "pswap": braket_instructions.PSwap(Parameter("theta")),
    "cv": braket_instructions.CV(),
    "ecr": qiskit_gates.ECRGate(),
    "prx": qiskit_gates.RGate(Parameter("theta"), Parameter("phi")),
    "iswap": qiskit_gates.iSwapGate(),
    "gpi": ionq_gates.GPIGate(Parameter("phi") / (2 * pi)),
    "gpi2": ionq_gates.GPI2Gate(Parameter("phi") / (2 * pi)),
    "ms": ionq_gates.MSGate(
        Parameter("phi0") / (2 * pi),
        Parameter("phi1") / (2 * pi),
        Parameter("theta") / (2 * pi),
    ),
    "gphase": qiskit_gates.GlobalPhaseGate(Parameter("theta")),
    "measure": qiskit_gates.Measure(),
    "unitary": qiskit_gates.UnitaryGate,
    "kraus": qiskit_qi.Kraus,
    "cc_prx": braket_instructions.CCPRx(
        Parameter("angle_1"), Parameter("angle_2"), Parameter("feedback_key")
    ),
    "measure_ff": braket_instructions.MeasureFF(Parameter("feedback_key")),
}

_BRAKET_VERBATIM_BOX_NAME = "verbatim"

_EPS = 1e-10  # global variable used to chop very small numbers to zero

_BRAKET_SUPPORTED_NOISES = [
    "kraus",
    "bitflip",
    "depolarizing",
    "amplitudedamping",
    "generalizedamplitudedamping",
    "phasedamping",
    "phaseflip",
    "paulichannel",
    "twoqubitdepolarizing",
    "twoqubitdephasing",
    # "twoqubitpaulichannel" no to_openqasm support yet
]

_PAULI_MAP = {
    "X": braket_observables.X,
    "Y": braket_observables.Y,
    "Z": braket_observables.Z,
}

_ADDITIONAL_U_GATES = {"u1", "u2", "u3"}

_SYMPY_FUNCTION_TO_QISKIT_METHOD = {
    sin: "sin",
    cos: "cos",
    tan: "tan",
    asin: "arcsin",
    acos: "arccos",
    atan: "arctan",
    exp: "exp",
    log: "log",
}

_TRANSPILER_GATE_SUBSTITUTES: dict[tuple[str, tuple[float | str, ...]], "QiskitInstruction"] = {
    ("rx", (pi,)): qiskit_gates.XGate(),
    ("rx", (-pi,)): qiskit_gates.XGate(),
    ("rx", (pi / 2,)): qiskit_gates.SXGate(),
    ("rx", (-pi / 2,)): qiskit_gates.SXdgGate(),
}

_BASIS_INVARIANT_RESULT_TYPES = (StateVector, DensityMatrix, Amplitude)
_Z_BASIS_RESULT_TYPES = (Probability,)
_OBSERVABLE_RESULT_TYPES = (Expectation, Sample, Variance)
_OBSERVABLE_TO_BASIS = {
    "z": "z",
    "i": "i",
    "x": "x",
    "y": "y",
    "h": "h",
}
_IDENTITY_BASIS = "i"
