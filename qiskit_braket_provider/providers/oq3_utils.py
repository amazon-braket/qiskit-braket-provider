"""Post-processing helpers for OpenQASM 3 emission targeting Amazon Braket.

These utilities transform the raw output of ``qiskit.qasm3.dumps()`` into
Braket-compatible OpenQASM 3 by renaming Qiskit gates to their Braket
equivalents, remapping virtual qubits to physical qubit labels, and normalizing
formatting to match Braket service conventions.
"""

import re
from collections.abc import Iterable, Sequence

from qiskit_braket_provider.providers.gate_mappings import (
    _GATE_RENAME_PATTERN,
    _QISKIT_TO_BRAKET_OQ3_NAMES,
)


def _post_process_oq3(
    oq3_source: str,
    qubit_labels: Sequence[int] | None,
    rename_gates: bool = True,
    output_names: Iterable[str] = (),
) -> str:
    """Post-process qasm3.dumps() output for Braket compatibility.

    Performs the following transformations:
    1. Renames Qiskit gate names to Braket-compatible OQ3 gate names (when needed)
    2. Remaps virtual qubit references to physical qubit labels
    3. Inserts ``#pragma braket verbatim`` before any ``box`` statements
    4. Replaces ``float[64]`` with ``float`` for parameter declarations
    5. Restores ``output`` keyword on classical register declarations whose name
       is listed in ``output_names``
    6. Strips indentation and removes empty lines (Braket convention)

    Args:
        oq3_source: Raw OpenQASM 3 string from ``qasm3.dumps()``.
        qubit_labels: Physical qubit indices for remapping. If ``None``,
            contiguous indices starting from 0 are assumed.
        rename_gates: Whether to rename Qiskit gate names to Braket equivalents.
            Set to ``False`` when the circuit already uses Braket-native gate names
            (e.g., after compiling to a device-native gate set).
        output_names: Names of classical registers that should be re-declared as
            OpenQASM ``output`` variables. Sourced from the compiled circuit's
            ``metadata["braket_output_variables"]``.

    Returns:
        The post-processed OpenQASM 3 string.
    """
    if rename_gates:
        oq3_source = _rename_gates(oq3_source)
    oq3_source = _remap_qubits(oq3_source, qubit_labels)
    oq3_source = _normalize_formatting(oq3_source, output_names)
    return oq3_source


def _rename_gates(oq3_source: str) -> str:
    """Replace Qiskit gate names with Braket-compatible names."""
    lines = oq3_source.split("\n")
    result = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("OPENQASM") or stripped.startswith("//"):
            result.append(line)
            continue
        result.append(_GATE_RENAME_PATTERN.sub(_gate_replacer, line))
    return "\n".join(result)


def _gate_replacer(match: re.Match) -> str:
    """Regex replacer callback for gate renaming."""
    return _QISKIT_TO_BRAKET_OQ3_NAMES[match.group(1)]


def _remap_qubits(oq3_source: str, qubit_labels: Sequence[int] | None) -> str:
    """Replace qubit references with physical qubit notation.

    Handles two cases produced by ``qasm3.dumps()``:

    1. **Layout-aware output** (``$N`` notation, no ``qubit[N] q;`` declaration):
       when the circuit's ``layout`` property is set, Qiskit emits physical
       qubit references directly. Remap ``$N`` → ``$qubit_labels[N]``.

    2. **Virtual register output** (``qubit[N] q; ... q[i]``): when no layout
       is attached, Qiskit falls back to virtual register syntax. Remove the
       ``qubit[N] q;`` declaration and remap ``q[i]`` → ``$qubit_labels[i]``.
    """
    if not qubit_labels:
        return oq3_source

    decl = re.compile(r"^qubit\[\d+\]\s+(\w+);\n?", re.MULTILINE)
    match = decl.search(oq3_source)
    if match:
        reg_name = match.group(1)
        oq3_source = decl.sub("", oq3_source, count=1)
        for i, label in enumerate(qubit_labels):
            oq3_source = oq3_source.replace(f"{reg_name}[{i}]", f"${label}")
        return oq3_source

    return re.sub(r"\$(\d+)", lambda m: f"${qubit_labels[int(m.group(1))]}", oq3_source)


def _normalize_formatting(oq3_source: str, output_names: Iterable[str] = ()) -> str:
    """Normalize OQ3 formatting to match Braket service conventions.

    - Strips leading indentation from all lines
    - Replaces ``float[64]`` with ``float`` in parameter declarations
    - Inserts ``#pragma braket verbatim`` before ``box`` statements
    - Removes space in ``box {`` → ``box{``
    - Restores ``output`` keyword on registers listed in ``output_names``
    - Removes empty lines
    """
    output_set = set(output_names)
    lines = []
    for line in oq3_source.split("\n"):
        line = line.lstrip()
        line = line.replace("float[64]", "float")
        if line == "box {":
            lines.append("#pragma braket verbatim")
            lines.append("box{")
            continue
        lines.append(_restore_output_declaration(line, output_set))
    return "\n".join(line for line in lines if line != "")


def _restore_output_declaration(line: str, output_names: set) -> str:
    """Prefix ``output`` onto a bit declaration ``qasm3.dumps`` rendered as plain.

    Qiskit models an output variable as an ordinary ``ClassicalRegister``, which
    ``qasm3.dumps`` renders as ``"bit[N] name;"``, so the keyword has to be put
    back on declarations whose name appears in ``output_names``.
    """
    if not line.startswith("bit["):
        return line
    name = line.removesuffix(";").rpartition(" ")[2]
    return f"output {line}" if name in output_names else line
