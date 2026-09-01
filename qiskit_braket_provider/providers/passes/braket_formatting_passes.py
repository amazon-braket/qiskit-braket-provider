"""Transpiler passes and string post-processing helpers for Braket-compatible OQ3 emission."""

import re
from collections.abc import Iterable, Sequence

from qiskit.circuit import ClassicalRegister, Instruction, Measure, QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from qiskit_braket_provider.providers.braket_annotations import (
    _BRAKET_VERBATIM_ANNOTATION_LINE,
    BraketVerbatimBox,
)
from qiskit_braket_provider.providers.gate_mappings import (
    _BRAKET_VERBATIM_PRAGMA_LINE,
    _OUTPUT_VARIABLES_KEY,
    _QISKIT_TO_BRAKET_SHIM,
    _SHIM_CLASSES,
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
        result.append(BraketVerbatimBox(inner), result.qubits, result.clbits)
        for instr in trailing_measurements:
            result.append(instr.operation, instr.qubits, instr.clbits)

        result.metadata = dag.metadata
        return circuit_to_dag(result)


def _shim_class_for(op: Instruction) -> type | None:
    """Return the shim class for ``op``, or ``None`` if no shim applies.

    Walks ``type(op).__mro__`` so ops that are Qiskit-generated singleton
    subclasses (e.g. ``_SingletonCXGate`` for ``CXGate``) match their mapped
    parent. Returns ``None`` if ``op`` is already a shim instance.
    """
    for cls in type(op).__mro__:
        if cls in _SHIM_CLASSES:
            return None
        if cls in _QISKIT_TO_BRAKET_SHIM:
            return _QISKIT_TO_BRAKET_SHIM[cls]
    return None


def _construct_shim(op: Instruction, shim_cls: type) -> Instruction:
    """Construct a shim instance mirroring ``op``'s parameters and label.

    Passing ``label`` at construction time (rather than mutating afterwards)
    yields a fresh, non-singleton instance from Qiskit's singleton machinery.
    """
    if op.label is not None:
        return shim_cls(*op.params, label=op.label)
    return shim_cls(*op.params)


def _rename_gates_in_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Return a new circuit with each renameable Qiskit gate replaced by its shim.

    Recurses into ``.blocks``-carrying ops (``IfElseOp`` / ``BoxOp`` / ...)
    at arbitrary nesting depth, so gates buried inside nested control-flow
    blocks are also renamed. Returns a fresh ``QuantumCircuit`` instead of
    mutating in place to avoid Qiskit's shared-block invariants.
    """
    new_circuit = circuit.copy_empty_like()
    for inst in circuit.data:
        op = inst.operation
        shim_cls = _shim_class_for(op)
        if shim_cls is not None:
            new_circuit.append(_construct_shim(op, shim_cls), inst.qubits, inst.clbits)
        elif getattr(op, "blocks", None):
            new_circuit.append(_rename_gates_in_block_op(op), inst.qubits, inst.clbits)
        else:
            new_circuit.append(op, inst.qubits, inst.clbits)
    return new_circuit


def _rename_gates_in_block_op(op: Instruction) -> Instruction:
    """Return a copy of a ``.blocks``-carrying op with each block's gates renamed."""
    return op.replace_blocks([_rename_gates_in_circuit(block) for block in op.blocks])


class RenameGates(TransformationPass):
    """Substitute Qiskit gates with Braket-named shim subclasses.

    ``qasm3.dumps`` reads ``Gate.name`` for serialization, so replacing e.g.
    ``CXGate`` with its shim (same semantics, ``name = "cnot"``) makes the
    serializer emit the Braket-native gate name. Recurses into
    ``.blocks``-carrying ops so nested gates are renamed too.
    """

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Substitute renameable Qiskit gates with their shim subclasses."""
        for node in dag.op_nodes():
            shim_cls = _shim_class_for(node.op)
            if shim_cls is not None:
                dag.substitute_node(node, _construct_shim(node.op, shim_cls))
            elif getattr(node.op, "blocks", None):
                dag.substitute_node(node, _rename_gates_in_block_op(node.op))
        return dag


def _remap_qubits(oq3_source: str, qubit_labels: Sequence[int] | None) -> str:
    """Replace qubit references with physical qubit notation.

    Handles two cases produced by ``qasm3.dumps()``:

    1. **Layout-aware output** (``$N`` notation, no ``qubit[N] q;`` declaration):
       when the circuit's ``layout`` property is set, Qiskit emits physical
       qubit references directly. Remap ``$N`` → ``$qubit_labels[N]``.

    2. **Virtual register output** (``qubit[N] q; ... q[i]``): when no layout
       is attached, Qiskit falls back to virtual register syntax. Remove the
       ``qubit[N] q;`` declaration and remap ``q[i]`` → ``$qubit_labels[i]``.
       This path primarily serves Qiskit circuits submitted with
       ``verbatim=True``, where physical ``$N`` output is required but no
       Qiskit ``.layout`` has been attached to the circuit.

    Raises:
        ValueError: If the source contains more than one ``qubit[N] ...;``
            declaration. Braketnsupports at most one quantum register.
        ValueError: If ``len(qubit_labels)`` does not match the ``N`` declared
            in the ``qubit[N] q;`` register (virtual-register case), or if a
            physical qubit reference ``$N`` in the source is out of range for
            ``qubit_labels`` (layout-aware case).
    """
    decl = re.compile(r"^qubit\[(\d+)\]\s+(\w+);\n?", re.MULTILINE)
    declarations = decl.findall(oq3_source)
    if len(declarations) > 1:
        raise ValueError(
            f"Braket does not support multiple quantum registers; "
            f"got {len(declarations)}: {[name for _, name in declarations]}."
        )

    if not qubit_labels:
        return oq3_source

    match = decl.search(oq3_source)
    if match:
        num_qubits = int(match.group(1))
        reg_name = match.group(2)
        if len(qubit_labels) != num_qubits:
            raise ValueError(
                f"qubit_labels length ({len(qubit_labels)}) does not match "
                f"the circuit's qubit count ({num_qubits})."
            )
        oq3_source = decl.sub("", oq3_source, count=1)
        for i, label in enumerate(qubit_labels):
            oq3_source = oq3_source.replace(f"{reg_name}[{i}]", f"${label}")
        return oq3_source

    referenced = [int(m) for m in re.findall(r"\$(\d+)", oq3_source)]
    if referenced and max(referenced) >= len(qubit_labels):
        raise ValueError(
            f"qubit_labels length ({len(qubit_labels)}) is too short for "
            f"physical qubit reference ${max(referenced)} in the source."
        )
    return re.sub(r"\$(\d+)", lambda m: f"${qubit_labels[int(m.group(1))]}", oq3_source)


def _normalize_formatting(oq3_source: str, output_names: Iterable[str] = ()) -> str:
    """Normalize OQ3 formatting to match Braket service conventions.

    - Strips leading indentation from all lines
    - Replaces ``float[64]`` with ``float``
    - Rewrites the Braket-verbatim annotation prefix emitted by ``qasm3.dumps``
      to Braket's canonical ``#pragma braket verbatim`` directive.
    - Restores ``output`` keyword on registers listed in ``output_names``
    - Removes empty lines
    """
    output_set = set(output_names)
    lines: list[str] = []
    for raw_line in oq3_source.split("\n"):
        line = raw_line.lstrip().replace("float[64]", "float")
        if _BRAKET_VERBATIM_ANNOTATION_LINE.match(line):
            lines.append(_BRAKET_VERBATIM_PRAGMA_LINE)
            continue
        lines.append(_restore_output_declaration(line, output_set))
    return "\n".join(line for line in lines if line != "")


def _restore_output_declaration(line: str, output_names: set[str]) -> str:
    """Re-add the ``output`` keyword to bit declarations whose name is in ``output_names``."""
    if not line.startswith("bit["):
        return line
    name = line.removesuffix(";").rpartition(" ")[2]
    return f"output {line}" if name in output_names else line
