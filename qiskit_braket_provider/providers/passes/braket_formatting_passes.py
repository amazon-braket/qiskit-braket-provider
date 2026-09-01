"""Transpiler passes for preparing circuits for Braket-compatible serialization."""

from qiskit.circuit import BoxOp, ClassicalRegister, Measure, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from qiskit_braket_provider.providers.gate_mappings import (
    _BRAKET_VERBATIM_BOX_NAME,
    _OUTPUT_VARIABLES_KEY,
)


def _plain_register_name(taken: set) -> str:
    """Return ``"b"``, or the first ``"b<i>"`` not shadowed by an output variable name."""
    if "b" not in taken:
        return "b"
    i = 0
    while f"b{i}" in taken:
        i += 1
    return f"b{i}"


class ConsolidateClbits(TransformationPass):
    """Expose every plain Clbit under a single ClassicalRegister named ``"b"``.

    "Plain" means any bit not declared ``output`` in the source program.
    Registers named in the circuit's ``"braket_output_variables"`` metadata
    (recorded by :func:`~qiskit_braket_provider.providers.adapter.to_qiskit` for
    OpenQASM ``output`` declarations) are kept as-is.

    The underlying Clbit objects are reused as the new register's members, so
    nothing on any DAGOpNode has to be rewired: Measure cargs and IfElseOp
    conditions continue to point at the same bits, which are now also members
    of ``"b"``. ``qasm3.dumps`` then emits ``bit[N] b;`` plus ``b[i]``
    references automatically.

    If ``"b"`` collides with an output variable name, the fallback is
    ``"b0"``, ``"b1"``, ... — whichever is first not shadowed.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Consolidate plain classical bits into a single register."""
        if not dag.clbits:
            return dag
        output_names = set((dag.metadata or {}).get(_OUTPUT_VARIABLES_KEY, {}))
        # Keep output-variable registers; drop the rest but keep their Clbits.
        kept_bits = set()
        for creg in list(dag.cregs.values()):
            if creg.name in output_names:
                kept_bits.update(creg)
            else:
                dag.remove_cregs(creg)
        plain = [bit for bit in dag.clbits if bit not in kept_bits]
        if plain:
            dag.add_creg(ClassicalRegister(name=_plain_register_name(output_names), bits=plain))
        return dag


class MoveMeasurementsToEnd(TransformationPass):
    """Reorder the DAG so all measurements appear at the end.

    Skips when the target device supports dynamic (mid-circuit) measurements,
    where measurement ordering carries semantic meaning and must not be
    rewritten.

    Args:
        dynamic_circuits_supported: If ``True``, the pass returns the DAG
            unchanged. Default: ``False``.
    """

    def __init__(self, dynamic_circuits_supported: bool = False):
        super().__init__()
        self._dynamic_circuits_supported = dynamic_circuits_supported

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Move every ``Measure`` op to the end of the DAG."""
        if self._dynamic_circuits_supported:
            return dag

        new_dag = dag.copy_empty_like()
        measurements = []
        for node in dag.topological_op_nodes():
            if isinstance(node.op, Measure):
                measurements.append(node)
            else:
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        for node in measurements:
            new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
        return new_dag


class WrapInVerbatimBox(TransformationPass):
    """Wrap operations in a ``BoxOp`` labeled ``"verbatim"``.

    The ``dynamic_circuits_supported`` flag mirrors the one on
    :class:`MoveMeasurementsToEnd`: it describes whether the target device
    can execute measurements that appear anywhere in the program.

    Args:
        dynamic_circuits_supported: If ``True``, every operation goes inside
            the verbatim box; measurement ordering is preserved. If ``False``
            (default), trailing measurements are placed outside the box.
            When ``False``, callers are expected to have run
            :class:`MoveMeasurementsToEnd` first (or otherwise guaranteed that
            all measurements are at the end of the circuit).
    """

    def __init__(self, dynamic_circuits_supported: bool = False):
        super().__init__()
        self._dynamic_circuits_supported = dynamic_circuits_supported

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Wrap operations in a verbatim ``BoxOp``."""
        circuit = dag_to_circuit(dag)
        inner = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        trailing_measurements = []

        for instr in circuit.data:
            is_measure = isinstance(instr.operation, Measure)
            if is_measure and not self._dynamic_circuits_supported:
                trailing_measurements.append(instr)
            else:
                inner.append(instr.operation, instr.qubits, instr.clbits)

        result = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        result.append(BoxOp(inner, label=_BRAKET_VERBATIM_BOX_NAME), result.qubits, result.clbits)
        for instr in trailing_measurements:
            result.append(instr.operation, instr.qubits, instr.clbits)

        result.metadata = dag.metadata
        return circuit_to_dag(result)
