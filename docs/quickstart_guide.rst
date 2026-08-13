################
Quickstart Guide
################

This guide shows you how to quickly start running Qiskit circuits on Amazon Braket devices using the
Qiskit-Braket provider.

****************
Running Circuits
****************

Running a circuit on an AWS simulator
=====================================

The following example runs a simple Qiskit circuit on the Amazon Braket SV1 state vector simulator.

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit_braket_provider import BraketProvider

    # Build a Bell pair.
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    # Select the Amazon Braket SV1 simulator and run.
    provider = BraketProvider()
    backend = provider.get_backend("SV1")
    job = backend.run(circuit, shots=100)
    result = job.result()
    print(result.get_counts())

For a list of available Amazon Braket simulators and their features, consult the `Amazon Braket
Developer Guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`_.

Running a circuit on a quantum hardware device
==============================================

To run a circuit on an Amazon Braket QPU, pass the name of the device to ``get_backend``. You can
list all available backends for your account (including simulators and QPUs) with
``provider.backends()``:

.. code-block:: python

    provider = BraketProvider()
    print(provider.backends())

A list of available quantum devices and their features can be found in the `Amazon Braket Developer
Guide <https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html>`_.

Running circuits on the local simulator
=======================================

The Qiskit-Braket provider also exposes the Amazon Braket local simulator, which runs on your own
machine and does not incur any Amazon Braket charges.

.. code-block:: python

    from qiskit_braket_provider import BraketLocalBackend

    backend = BraketLocalBackend()
    job = backend.run(circuit, shots=100)
    result = job.result()
    print(result.get_counts())

*****************
More information
*****************

For more examples, see the :doc:`how-to guides <how_tos/index>` and the :doc:`tutorials
<tutorials/index>`.
