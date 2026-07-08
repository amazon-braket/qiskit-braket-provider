# Changelog

## v0.17.5 (2026-07-08)

### Bug Fixes and Other Changes

 * register Braket-native two-qubit gates in to_qiskit

## v0.17.4 (2026-07-06)

### Bug Fixes and Other Changes

 * correctly write measures across multiple bit registers in to_qiskit

## v0.17.3 (2026-06-30)

### Bug Fixes and Other Changes

 * update tests & tutorials to current Braket devices

## v0.17.2 (2026-06-29)

### Bug Fixes and Other Changes

 * allocate missing qubits in add_measure for physical-qubit refs
 * Preserve if/measure inside verbatim box in to_qiskit
 * Share parent bits in control-flow body circuits in to_qiskit

## v0.17.1 (2026-06-25)

### Bug Fixes and Other Changes

 * classical bit in box

## v0.17.0 (2026-06-22)

### Features

 * add abelian grouping of commuting observables to BraketEstimator
 * add verbatim support from Circuits to QuantumCircuit

## v0.16.0 (2026-06-17)

### Features

 * add device emulator support

### Bug Fixes and Other Changes

 * Don't use qubit labels for virtual qubits
 * allow empty circuit in _get_circuits

## v0.15.0 (2026-06-16)

### Features

 * support Braket parameter functions in to_qiskit

## v0.14.4 (2026-05-19)

### Bug Fixes and Other Changes

 * use new default simulator interface

## v0.14.3 (2026-05-06)

### Bug Fixes and Other Changes

 * fixed missing layout after _substitute call

## v0.14.2 (2026-04-29)

### Bug Fixes and Other Changes

 * unify MCM-dependency check in _QiskitProgramContext
 * missing layout bug

## v0.14.1 (2026-04-29)

### Bug Fixes and Other Changes

 * Ignore barriers if unsupported

## v0.14.0 (2026-04-27)

### Features

 * adding loop support for OpenQASM

### Bug Fixes and Other Changes

 * remove pylint config and stale pylint comments

## v0.13.1 (2026-04-23)

### Bug Fixes and Other Changes

 * fix / feature: Add barrier to Target
 * add more ruff checks and fix raised errors
 * adjust coupling_map logic
 * Fix documentation wording in 4_tutorial_native_programming.ipynb
