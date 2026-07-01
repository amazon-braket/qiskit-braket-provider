"""Transpiler passes for preserving verbatim boxes through compilation."""

from qiskit.circuit import Barrier, BoxOp, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass

from qiskit_braket_provider.providers.gate_mappings import _BRAKET_VERBATIM_BOX_NAME


def _indexed_label(base: str, index: int) -> str:
    return f"{base}__{index}"


def _is_verbatim_label(label: str | None, base: str) -> bool:
    """Check if a label is a verbatim barrier label (base or indexed)."""
    if label is None:
        return False
    return label == base or (label.startswith(f"{base}__") and label[len(base) + 2 :].isdigit())


class ExtractVerbatimBoxes(TransformationPass):
    """Swap verbatim ``BoxOp`` nodes for labeled barriers before transpilation.

    The original box circuits are stashed in ``property_set["verbatim_boxes"]``
    as a dict mapping indexed labels to ``QuantumCircuit`` instances for
    :class:`RestoreVerbatimBoxes` to restore afterwards.

    Args:
        verbatim_box_name: Label used to identify verbatim ``BoxOp`` nodes.
    """

    def __init__(self, verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME):
        super().__init__()
        self._verbatim_box_name = verbatim_box_name

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Replace matching ``BoxOp`` nodes with labeled barriers.

        Raises:
            ValueError: If the DAG already contains a barrier whose label
                matches ``verbatim_box_name``.
        """
        for node in dag.topological_op_nodes():
            if isinstance(node.op, Barrier) and _is_verbatim_label(
                getattr(node.op, "label", None), self._verbatim_box_name
            ):
                raise ValueError(
                    f"Circuit contains a Barrier with label '{node.op.label}' "
                    "which conflicts with the verbatim box label"
                )

        verbatim_boxes = {}
        index = 0
        for node in dag.topological_op_nodes():
            if not isinstance(node.op, BoxOp):
                continue
            if getattr(node.op, "label", None) != self._verbatim_box_name:
                continue
            label = _indexed_label(self._verbatim_box_name, index)
            verbatim_boxes[label] = (node.op.blocks[0], [dag.find_bit(c).index for c in node.cargs])
            barrier = Barrier(len(node.qargs), label=label)
            if node.cargs:
                # BoxOp has clbits — substitute_node requires matching width,
                # so replace with a sub-DAG containing only the barrier on qubits.
                qc = QuantumCircuit(len(node.qargs), len(node.cargs))
                qc.append(barrier, list(range(len(node.qargs))))
                dag.substitute_node_with_dag(node, circuit_to_dag(qc))
            else:
                dag.substitute_node(node, barrier)
            index += 1

        self.property_set["verbatim_boxes"] = verbatim_boxes
        return dag


class RestoreVerbatimBoxes(TransformationPass):
    """Replace labeled barriers with the original verbatim gate sequences.

    Reads ``property_set["verbatim_boxes"]`` populated by
    :class:`ExtractVerbatimBoxes`.

    Args:
        verbatim_box_name: Label used to identify placeholder barriers.
    """

    def __init__(self, verbatim_box_name: str = _BRAKET_VERBATIM_BOX_NAME):
        super().__init__()
        self._verbatim_box_name = verbatim_box_name

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Splice stashed gate sequences back in place of labeled barriers.

        Raises:
            RuntimeError: If a labeled barrier has no matching stashed box.
            RuntimeError: If stashed boxes remain unconsumed after processing.
        """
        verbatim_boxes = self.property_set.get("verbatim_boxes", {})
        if not verbatim_boxes:
            return dag

        for node in dag.topological_op_nodes():
            if not isinstance(node.op, Barrier):
                continue
            label = getattr(node.op, "label", None)
            if not _is_verbatim_label(label, self._verbatim_box_name):
                continue
            if label not in verbatim_boxes:
                raise RuntimeError(
                    f"Internal error: verbatim barrier '{label}' has no matching box. "
                    f"This is a bug in the verbatim pass pipeline."
                )
            box_circuit, clbit_indices = verbatim_boxes.pop(label)
            box_dag = circuit_to_dag(box_circuit)
            if clbit_indices:
                # Build a replacement DAG that includes the box's clbits.
                qubit_map = dict(zip(box_dag.qubits, node.qargs, strict=True))
                clbit_map = dict(
                    zip(
                        box_dag.clbits,
                        [dag.clbits[i] for i in clbit_indices],
                        strict=True,
                    )
                )
                dag.substitute_node_with_dag(node, box_dag, wires={**qubit_map, **clbit_map})
            else:
                dag.substitute_node_with_dag(node, box_dag)

        if verbatim_boxes:
            raise RuntimeError(
                f"Internal error: stashed verbatim boxes were not restored "
                f"(no matching barriers found): {list(verbatim_boxes.keys())}. "
                f"Their barriers may have been removed during transpilation."
            )
        return dag
