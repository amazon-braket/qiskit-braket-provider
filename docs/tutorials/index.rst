Tutorials
=========

These notebooks are also published and integration-tested in the
`Amazon Braket Examples <https://github.com/amazon-braket/amazon-braket-examples>`_ repository
under ``examples/qiskit/``. When updating a tutorial here, keep the corresponding copy in sync:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Provider tutorial
     - Examples repository notebook
   * - :doc:`0_tutorial_qiskit-braket-provider_overview`
     - `2_Overview_of_the_Qiskit_Braket_provider.ipynb <https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/qiskit/2_Overview_of_the_Qiskit_Braket_provider.ipynb>`_
   * - :doc:`1_tutorial_vqe`
     - `3_Running_VQE_on_Braket.ipynb <https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/qiskit/3_Running_VQE_on_Braket.ipynb>`_
   * - :doc:`2_tutorial_hybrid_jobs`
     - `4_Hybrid_Jobs_with_Qiskit.ipynb <https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/qiskit/4_Hybrid_Jobs_with_Qiskit.ipynb>`_
   * - :doc:`3_tutorial_minimum_eigen_optimizer`
     - `5_Minimum_Eigen_Optimizer.ipynb <https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/qiskit/5_Minimum_Eigen_Optimizer.ipynb>`_
   * - :doc:`4_tutorial_native_programming`
     - `6_Native_Programming.ipynb <https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/qiskit/6_Native_Programming.ipynb>`_
   * - :doc:`5_tutorial_transpilation`
     - `7_Transpilation.ipynb <https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/qiskit/7_Transpilation.ipynb>`_
   * - :doc:`6_tutorial_primitives`
     - `8_Braket_Native_Primitives.ipynb <https://github.com/amazon-braket/amazon-braket-examples/blob/main/examples/qiskit/8_Braket_Native_Primitives.ipynb>`_

Run notebook integration tests from the examples repository (see `TESTING.md <https://github.com/amazon-braket/amazon-braket-examples/blob/main/TESTING.md>`_):

.. code-block:: bash

   pytest test/integ_tests/test_all_notebooks.py -k "qiskit-2 or qiskit-3 or qiskit-4 or qiskit-5 or qiskit-6 or qiskit-7 or qiskit-8"

.. toctree::
   :glob:
   :maxdepth: 1

   *
