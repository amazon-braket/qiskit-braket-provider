"""Transpiler passes for preparing circuits for OpenQASM 3 serialization."""

from qiskit.circuit import BoxOp, ClassicalRegister, Measure, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from qiskit_braket_provider.providers.gate_mappings import _BRAKET_VERBATIM_BOX_NAME

_OUTPUT_VARIABLES_KEY = "braket_output_variables"


class ConsolidateClbits(TransformationPass):
    """Expose every plain Clbit under a single ClassicalRegister named ``"b"``.

    "Plain" means any bit not declared ``output`` in the source program.
    Registers named in the circuit's ``"braket_output_variables"`` metadata
    are kept as-is so ``_post_process_oq3`` can re-emit them as
    ``output bit[N] name;`` declarations.

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


def _plain_register_name(taken: set) -> str:
    """Return ``"b"``, or the first ``"b<i>"`` not shadowed by an output variable name."""
    if "b" not in taken:
        return "b"
    i = 0
    while f"b{i}" in taken:
        i += 1
    return f"b{i}"


class WrapInVerbatimBox(TransformationPass):
    """Wrap circuit operations in a verbatim BoxOp for ``#pragma braket verbatim``.

    All non-measurement operations are collected into an inner circuit and wrapped
    in a :class:`~qiskit.circuit.BoxOp` with a verbatim label. Measurements are
    placed after the box.

    Preserves ``dag.metadata`` on the resulting DAG so downstream passes and
    serializers still see the input circuit's metadata (notably
    ``braket_output_variables``).

    Args:
        verbatim_box_name: Label for the BoxOp. Default: ``"verbatim"``.
    """

    def __init__(self, verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME):
        super().__init__()
        self._verbatim_box_name = verbatim_box_name

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Wrap non-measurement operations in a verbatim BoxOp."""
        circuit = dag_to_circuit(dag)
        num_qubits = circuit.num_qubits

        inner = QuantumCircuit(num_qubits, circuit.num_clbits)
        measurements = []

        for instr in circuit.data:
            if isinstance(instr.operation, Measure):
                measurements.append(instr)
            else:
                qubit_indices = [circuit.find_bit(q).index for q in instr.qubits]
                clbit_indices = [circuit.find_bit(c).index for c in instr.clbits]
                inner.append(
                    instr.operation,
                    [inner.qubits[i] for i in qubit_indices],
                    [inner.clbits[i] for i in clbit_indices],
                )

        result = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        box_op = BoxOp(inner, label=self._verbatim_box_name)
        result.append(box_op, result.qubits, result.clbits)

        for instr in measurements:
            qubit_indices = [circuit.find_bit(q).index for q in instr.qubits]
            clbit_indices = [circuit.find_bit(c).index for c in instr.clbits]
            result.append(
                instr.operation,
                [result.qubits[i] for i in qubit_indices],
                [result.clbits[i] for i in clbit_indices],
            )

        result.metadata = dag.metadata
        return circuit_to_dag(result)
