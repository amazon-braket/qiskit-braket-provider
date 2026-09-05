"""Braket-verbatim annotation and box-op infrastructure for OpenQASM 3 emission.

This module hosts the Qiskit :class:`~qiskit.circuit.annotation.Annotation` and
:class:`~qiskit.circuit.BoxOp` machinery used to mark verbatim blocks in Qiskit
circuits so that ``qasm3.dumps`` emits Braket's ``#pragma braket verbatim``
directive.
"""

import re

from qiskit.circuit import BoxOp, QuantumCircuit
from qiskit.circuit.annotation import Annotation, OpenQASM3Serializer

from qiskit_braket_provider.providers.gate_mappings import (
    _BRAKET_VERBATIM_BOX_NAME,
    _BRAKET_VERBATIM_PRAGMA_PAYLOAD,
)

_BRAKET_VERBATIM_ANNOTATION_LINE = re.compile(r"^@braket_verbatim\s+pragma\s+braket\s+verbatim\s*$")


class BraketVerbatim(Annotation):
    """Marks a ``BoxOp`` as a Braket verbatim block for OpenQASM 3 emission."""

    namespace = "braket_verbatim"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BraketVerbatim)

    def __hash__(self) -> int:
        return hash(BraketVerbatim)


class BraketVerbatimSerializer(OpenQASM3Serializer):
    """Serializes :class:`BraketVerbatim` as ``pragma braket verbatim``."""

    def dump(self, annotation: Annotation) -> str:
        if isinstance(annotation, BraketVerbatim):
            return _BRAKET_VERBATIM_PRAGMA_PAYLOAD
        return NotImplemented  # pragma: no cover

    def load(self, _namespace: str, _payload: str) -> Annotation:
        return NotImplemented  # pragma: no cover


_BRAKET_ANNOTATION_HANDLERS: dict[str, OpenQASM3Serializer] = {
    BraketVerbatim.namespace: BraketVerbatimSerializer(),
}


class BraketVerbatimBox(BoxOp):
    """A ``BoxOp`` pre-configured as a Braket verbatim block.

    Attaches :class:`BraketVerbatim` and sets ``label=_BRAKET_VERBATIM_BOX_NAME``
    (overridable via ``label``) so ``qasm3.dumps`` emits the verbatim directive.
    """

    def __init__(self, body: QuantumCircuit, label: str = _BRAKET_VERBATIM_BOX_NAME) -> None:
        super().__init__(
            body,
            label=label,
            annotations=[BraketVerbatim()],
        )
