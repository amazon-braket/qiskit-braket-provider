"""Device target construction for Qiskit backends.

This module provides functions for converting Braket device properties into
Qiskit Target objects, including gate sets, connectivity, and parameter restrictions.
"""

import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping
from math import inf, pi
from typing import Self, TypeAlias

import qiskit.circuit.library as qiskit_gates
from qiskit import QuantumCircuit
from qiskit.circuit import (
    Barrier,
    Gate,
    Measure,
    Parameter,
)
from qiskit.circuit import Instruction as QiskitInstruction
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import (
    InstructionProperties,
    PassManager,
    QubitProperties,
    Target,
    TransformationPass,
)

from braket.aws import AwsDevice, AwsDeviceType
from braket.device_schema import (
    DeviceActionType,
    DeviceCapabilities,
    OpenQASMDeviceActionProperties,
)
from braket.device_schema.ionq import IonqDeviceCapabilities
from braket.device_schema.rigetti import RigettiDeviceCapabilities, RigettiDeviceCapabilitiesV2
from braket.device_schema.simulators import GateModelSimulatorDeviceCapabilities
from braket.device_schema.standardized_gate_model_qpu_device_properties_v1 import (
    StandardizedGateModelQpuDeviceProperties as StandardizedPropertiesV1,
)
from braket.device_schema.standardized_gate_model_qpu_device_properties_v2 import (
    StandardizedGateModelQpuDeviceProperties as StandardizedPropertiesV2,
)
from braket.devices import Device, LocalSimulator
from braket.ir.openqasm.modifiers import Control
from braket.parametric import FreeParameter, Parameterizable
from qiskit_braket_provider.exception import QiskitBraketException
from qiskit_braket_provider.providers.gate_mappings import (
    _BRAKET_GATE_NAME_TO_QISKIT_GATE,
    _BRAKET_TO_QISKIT_NAMES,
    _CONTROLLED_GATES_BY_QUBIT_COUNT,
    _STANDARD_GATE_NAME_MAPPING,
)

_ADDITIONAL_U_GATES = {"u1", "u2", "u3"}

_TRANSPILER_GATE_SUBSTITUTES: dict[tuple[str, tuple[float | str, ...]], Gate] = {
    ("rx", (pi,)): qiskit_gates.XGate(),
    ("rx", (-pi,)): qiskit_gates.XGate(),
    ("rx", (pi / 2,)): qiskit_gates.SXGate(),
    ("rx", (-pi / 2,)): qiskit_gates.SXdgGate(),
}

_ParamKey: TypeAlias = tuple[float | str, ...]
_QubitSet: TypeAlias = set[tuple[int, ...]]
_ParameterRestrictions: TypeAlias = dict[str, dict[_ParamKey, _QubitSet]]


class SubstitutedTarget(Target):
    """A transpiler target that applies qubit-specific gate substitutions after transpilation.

    This extends Qiskit's :class:`~qiskit.transpiler.Target` to support devices where
    certain gates must be replaced with hardware-specific equivalents on particular qubits
    (e.g. IonQ's GPi/GPi2/MS gate decompositions). The substitutions are applied as a
    post-transpilation pass.
    """

    def __new__(cls, *args, **kwargs) -> Self:
        out = super().__new__(cls, *args, **kwargs)
        gate_substitutes: dict[str, dict[tuple[int, ...], QiskitInstruction]] = {}
        out._gate_substitutes = gate_substitutes
        out._pass_manager = PassManager([_SubstituteGates(gate_substitutes)])
        return out

    def _substitute(
        self, circuits: QuantumCircuit | Iterable[QuantumCircuit]
    ) -> QuantumCircuit | list[QuantumCircuit]:
        single = isinstance(circuits, QuantumCircuit)
        circuit_list: list[QuantumCircuit] = [circuits] if single else list(circuits)
        results: list[QuantumCircuit] = self._pass_manager.run(circuit_list)
        for original, result in zip(circuit_list, results, strict=True):
            if original._layout is not None:
                result._layout = original._layout
        return results[0] if single else results


class _SubstituteGates(TransformationPass):
    def __init__(
        self, gate_substitutes: Mapping[str, Mapping[tuple[int, ...], QiskitInstruction]]
    ) -> None:
        super().__init__()
        self._gate_substitutes = gate_substitutes

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        if not self._gate_substitutes:
            return dag
        qubits = {q: i for i, q in enumerate(dag.qubits)}
        for node in dag.op_nodes():
            if (op_name := node.op.name) in self._gate_substitutes:
                dag.substitute_node(
                    node, self._gate_substitutes[op_name][tuple(qubits[q] for q in node.qargs)]
                )
        return dag


def native_gate_connectivity(properties: DeviceCapabilities) -> list[list[int]] | None:
    """Returns the connectivity natively supported by a Braket device from its properties

    Args:
        properties (DeviceCapabilities): The device properties of the Braket device.

    Returns:
        list[list[int]] | None: A list of connected qubit pairs or ``None``
        if the device is fully connected.
    """
    device_connectivity = properties.paradigm.connectivity
    return (
        [
            [int(x), int(y)]
            for x, neighborhood in device_connectivity.connectivityGraph.items()
            for y in neighborhood
        ]
        if not device_connectivity.fullyConnected
        else None
    )


def native_gate_set(properties: DeviceCapabilities) -> set[str]:
    """Returns the gate set natively supported by a Braket device from its properties

    Args:
        properties (DeviceCapabilities): The device properties of the Braket device.

    Returns:
        set[str]: The names of Qiskit gates natively supported by the Braket device.
    """
    native_list = properties.paradigm.nativeGateSet
    return {
        _BRAKET_TO_QISKIT_NAMES[op.lower()]
        for op in native_list
        if op.lower() in _BRAKET_TO_QISKIT_NAMES
    }


def native_angle_restrictions(
    properties: DeviceCapabilities,
) -> dict[str, dict[int, set[float] | tuple[float, float]]]:
    """Returns angle restrictions for gates natively supported by a Braket device.

    The returned mapping specifies, for each gate name, constraints on the
    gate parameters indexed by their position. The constraint can either be a
    set of allowed angles or a tuple representing an inclusive ``(min, max)``
    range. Angle units are in radians.

    Args:
        properties (DeviceCapabilities): The device properties of the Braket device.

    Returns:
        dict[str, dict[int, set[float] | tuple[float, float]]]: Mapping
        of gate names to parameter index restrictions.
    """

    if isinstance(properties, (RigettiDeviceCapabilities, RigettiDeviceCapabilitiesV2)):
        return {"rx": {0: {pi, -pi, pi / 2, -pi / 2}}}
    if isinstance(properties, IonqDeviceCapabilities):
        return {"ms": {2: (0.0, 0.25)}}
    return {}


def gateset_from_properties(properties: OpenQASMDeviceActionProperties) -> set[str]:
    """Returns the gateset supported by a Braket device with the given properties

    Args:
        properties (OpenQASMDeviceActionProperties): The action properties of the Braket device.

    Returns:
        set[str]: The names of the gates supported by the device
    """
    gateset = {
        _BRAKET_TO_QISKIT_NAMES[op.lower()]
        for op in properties.supportedOperations
        if op.lower() in _BRAKET_TO_QISKIT_NAMES
    }
    if "u" in gateset:
        gateset.update(_ADDITIONAL_U_GATES)
    max_control = 0
    for modifier in properties.supportedModifiers:
        if isinstance(modifier, Control):
            max_control = modifier.max_qubits
            break
    return gateset.union(_get_controlled_gateset(gateset, max_control))


def _get_controlled_gateset(base_gateset: set[str], max_qubits: int | None = None) -> set[str]:
    """Returns the Qiskit gates expressible as controlled versions of existing Braket gates

    This set can be filtered by the maximum number of control qubits.

    Args:
        base_gateset (set[str]): The base (without control modifiers) gates supported
        max_qubits (int | None): The maximum number of control qubits that can be used to express
            the Qiskit gate as a controlled Braket gate. If ``None``, then there is no limit to the
            number of control qubits. Default: ``None``.

    Returns:
        set[str]: The names of the controlled gates.
    """
    max_control = max_qubits if max_qubits is not None else inf
    return {
        controlled_gate
        for control_count, gate_map in _CONTROLLED_GATES_BY_QUBIT_COUNT.items()
        for controlled_gate, base_gate in gate_map.items()
        if control_count <= max_control and base_gate in base_gateset
    }


def local_simulator_to_target(simulator: LocalSimulator) -> Target:
    """Converts properties of a Braket LocalSimulator into a Qiskit Target object.

    Args:
        simulator (LocalSimulator): Amazon Braket ``LocalSimulator``

    Returns:
        Target: Target for Qiskit backend
    """
    return _simulator_target(
        simulator, f"Target for Amazon Braket local simulator: {simulator.name}"
    )


def aws_device_to_target(device: AwsDevice) -> Target:
    """Converts properties of Braket AwsDevice into a Qiskit Target object.

    Args:
        device (AwsDevice): Amazon Braket ``AwsDevice``

    Returns:
        Target: Target for Qiskit backend
    """
    match device.type:
        case AwsDeviceType.QPU:
            return _qpu_target(device, f"Target for Amazon Braket QPU: {device.name}")
        case AwsDeviceType.SIMULATOR:
            return _simulator_target(device, f"Target for Amazon Braket simulator: {device.name}")
    raise QiskitBraketException(
        "Cannot convert to target. "
        f"{device.properties.__class__} device capabilities are not supported."
    )


def _simulator_target(device: Device, description: str) -> Target:
    properties: GateModelSimulatorDeviceCapabilities = device.properties
    target = Target(description=description, num_qubits=properties.paradigm.qubitCount)
    action = properties.action.get(DeviceActionType.OPENQASM) or properties.action.get(
        DeviceActionType.JAQCD
    )
    for operation in action.supportedOperations:
        instruction = _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(operation.lower())
        if instruction:
            target.add_instruction(instruction, name=_BRAKET_TO_QISKIT_NAMES[operation.lower()])
    if isinstance(action, OpenQASMDeviceActionProperties):
        max_control = 0
        for modifier in action.supportedModifiers:
            if isinstance(modifier, Control):
                max_control = modifier.max_qubits
                break
        for gate in _get_controlled_gateset(target.keys(), max_control):
            if gate in _STANDARD_GATE_NAME_MAPPING:
                target.add_instruction(_STANDARD_GATE_NAME_MAPPING[gate])
    target.add_instruction(Measure())
    target.add_instruction(Barrier(1))
    return target


def _qpu_target(device: AwsDevice, description: str) -> Target:
    properties: DeviceCapabilities = device.properties
    topology = device.topology_graph
    standardized = properties.standardized
    indices = {q: i for i, q in enumerate(sorted(topology.nodes))}

    qubit_properties = []
    default_instruction_props = InstructionProperties(error=0)
    instruction_props_measurement: dict[tuple[int, ...], InstructionProperties] = {}
    instruction_props_1q: dict[tuple[int, ...], InstructionProperties] = {}
    instruction_props_2q: dict[str, dict[tuple[int, ...], InstructionProperties]] = {}
    # TODO: Support V3 standardized properties
    if isinstance(standardized, (StandardizedPropertiesV1, StandardizedPropertiesV2)):
        props_1q = standardized.oneQubitProperties
        for q in sorted(int(q) for q in props_1q):
            if q not in indices:
                warnings.warn(
                    f"Qubit {q} found in device properties but not in topology. "
                    f"Skipping qubit {q} and its associated properties.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            props = props_1q[str(q)]
            key = (indices[q],)
            for fidelity in props.oneQubitFidelity:
                match fidelity.fidelityType.name.lower():
                    case "readout":
                        instruction_props_measurement[key] = InstructionProperties(
                            # Use highest known error rate
                            error=max(
                                1 - fidelity.fidelity,
                                instruction_props_measurement.get(
                                    key, default_instruction_props
                                ).error,
                            )
                        )
                    case name if "readout_error" not in name:
                        instruction_props_1q[key] = InstructionProperties(
                            error=max(
                                1 - fidelity.fidelity,
                                instruction_props_1q.get(key, default_instruction_props).error,
                            )
                        )
            qubit_properties.append(QubitProperties(t1=props.T1.value, t2=props.T2.value))
        instruction_props_2q.update(
            _build_instruction_props_2q(standardized, indices, default_instruction_props)
        )

    default_props_1q: dict[tuple[int, ...], None] = {(i,): None for i in indices.values()}
    default_props_2q: dict[tuple[int, ...], None] = {
        (indices[u], indices[v]): None for u, v in topology.edges
    }
    if not instruction_props_measurement:
        instruction_props_measurement.update(default_props_1q)
    if not instruction_props_1q:
        instruction_props_1q.update(default_props_1q)

    parameter_restrictions = _get_parameter_restrictions(device, indices)
    target = SubstitutedTarget(
        description=description,
        num_qubits=len(qubit_properties or indices),
        qubit_properties=qubit_properties or None,
    )
    if parameter_restrictions:
        _add_instructions_parameter_restrictions(
            target,
            parameter_restrictions,
            instruction_props_1q,
            instruction_props_2q,
            default_props_2q,
        )
    else:
        _add_instructions_no_parameter_restrictions(
            target,
            properties.paradigm.nativeGateSet,
            instruction_props_1q,
            instruction_props_2q,
            default_props_2q,
        )

    if "barrier" in properties.paradigm.nativeGateSet:
        target.add_instruction(Barrier(1))

    # Add measurement if not already added
    if "measure" not in target:
        target.add_instruction(Measure(), instruction_props_measurement)
    return target


def _build_instruction_props_2q(
    standardized: StandardizedPropertiesV1 | StandardizedPropertiesV2,
    indices: Mapping[int, int],
    default_properties: InstructionProperties,
) -> dict[str, dict[tuple[int, ...], InstructionProperties]]:
    instruction_props_2q: dict[str, dict[tuple[int, ...], InstructionProperties]] = defaultdict(
        dict
    )
    for k, props in standardized.twoQubitProperties.items():
        qubits = [int(q) for q in k.split("-")]
        # Check if all qubits in the edge exist in topology
        if not all(q in indices for q in qubits):
            missing_qubits = [q for q in qubits if q not in indices]
            warnings.warn(
                f"Edge {k} contains qubits {missing_qubits} not found in topology. "
                f"Skipping edge {k} and its associated properties.",
                UserWarning,
                stacklevel=2,
            )
            continue

        for fidelity in props.twoQubitGateFidelity:
            if gate_name := _BRAKET_TO_QISKIT_NAMES.get(fidelity.gateName.lower()):
                edge = tuple(indices[q] for q in qubits)
                instruction_props_2q[gate_name][edge] = InstructionProperties(
                    error=max(
                        1 - fidelity.fidelity,
                        instruction_props_2q[gate_name].get(edge, default_properties).error,
                    )
                )
    # Standardized 2q gate props assume bidirectionality
    for edge_props in instruction_props_2q.values():
        edge_props.update({
            tuple(reversed(edge)): instruction_props
            for edge, instruction_props in edge_props.items()
        })
    return instruction_props_2q


def _get_parameter_restrictions(
    device: AwsDevice, qubit_indices: Mapping[int, int]
) -> _ParameterRestrictions:
    cal = device.gate_calibrations
    parameter_restrictions: _ParameterRestrictions = defaultdict(lambda: defaultdict(set))
    for gate, target in cal.pulse_sequences if cal else {}:
        gate_name = gate.name.lower()
        qubits = tuple(qubit_indices[q] for q in target)
        if isinstance(gate, Parameterizable):
            param_key = tuple(
                param.name if isinstance(param, FreeParameter) else param
                for param in gate.parameters
            )
            parameter_restrictions[gate_name][tuple(param_key)].add(qubits)
        else:
            parameter_restrictions[gate_name][()].add(qubits)
    return parameter_restrictions


def _add_instructions_parameter_restrictions(
    target: SubstitutedTarget,
    parameter_restrictions: Mapping[str, Mapping[_ParamKey, _QubitSet]],
    instruction_props_1q: Mapping[tuple[int, ...], InstructionProperties],
    instruction_props_2q: Mapping[str, Mapping[tuple[int, ...], InstructionProperties]],
    default_props_2q: Mapping[tuple[int, ...], InstructionProperties | None],
) -> None:
    for braket_name, restrictions in parameter_restrictions.items():
        if instruction := _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(braket_name):
            gate_name = instruction.name
            match num_qubits := instruction.num_qubits:
                case 1:
                    _add_single_instruction_parameter_restriction(
                        target,
                        instruction,
                        braket_name,
                        restrictions,
                        instruction_props_1q,
                    )
                case 2:
                    _add_single_instruction_parameter_restriction(
                        target,
                        instruction,
                        braket_name,
                        restrictions,
                        instruction_props_2q.get(gate_name, default_props_2q),
                    )
                case _:
                    warnings.warn(
                        f"Instruction {gate_name} has {num_qubits} qubits "
                        "and cannot be added to target",
                        stacklevel=2,
                    )


def _add_single_instruction_parameter_restriction(
    target: SubstitutedTarget,
    instruction: QiskitInstruction,
    braket_name: str,
    restrictions: Mapping[_ParamKey, _QubitSet],
    gate_properties: Mapping[tuple[int, ...], InstructionProperties],
) -> None:
    for restriction, qubits in restrictions.items():
        props = {q: props for q, props in gate_properties.items() if q in qubits}
        instruction_copy = instruction.copy()
        if restriction:
            instruction_copy.params = [
                Parameter(param) if isinstance(param, str) else param for param in restriction
            ]
        if substitute := _TRANSPILER_GATE_SUBSTITUTES.get((braket_name, restriction)):
            substitute_name = substitute.name
            if instruction_target := target.get(substitute_name):
                # Nothing to do with Hamiltonian mechanics; variable names are coincidental :)
                for q, p in props.items():
                    if not (current := instruction_target.get(q)) or current.error > p.error:
                        instruction_target[q] = p
                        target._gate_substitutes[substitute_name][q] = instruction_copy
            else:
                target.add_instruction(substitute, props)
                target._gate_substitutes[substitute_name] = dict.fromkeys(props, instruction_copy)
        else:
            target.add_instruction(instruction_copy, props)


def _add_instructions_no_parameter_restrictions(
    target: SubstitutedTarget,
    native_gateset: set[str],
    instruction_props_1q: Mapping[tuple[int, ...], InstructionProperties],
    instruction_props_2q: Mapping[str, Mapping[tuple[int, ...], InstructionProperties]],
    default_props_2q: Mapping[tuple[int, ...], InstructionProperties | None],
) -> None:
    for operation in native_gateset:
        if instruction := _BRAKET_GATE_NAME_TO_QISKIT_GATE.get(operation.lower()):
            gate_name = instruction.name
            match num_qubits := instruction.num_qubits:
                case 1:
                    target.add_instruction(instruction, instruction_props_1q)
                case 2:
                    target.add_instruction(
                        instruction, instruction_props_2q.get(gate_name, default_props_2q)
                    )
                case _:
                    warnings.warn(
                        f"Instruction {gate_name} has {num_qubits} qubits "
                        "and cannot be added to target",
                        stacklevel=2,
                    )
