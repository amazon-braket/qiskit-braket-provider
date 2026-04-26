####################################################
Getting Started with the Qiskit-Braket Provider
####################################################

The Qiskit-Braket provider allows you to run Qiskit programs on Amazon Braket devices and
simulators. You can get started using an Amazon Braket notebook instance or using your own
environment.

For more information about Amazon Braket, see the full set of documentation at
https://docs.aws.amazon.com/braket/index.html.

************************************************
Getting started using an Amazon Braket notebook
************************************************

You can use the AWS Console to enable Amazon Braket and create an Amazon Braket notebook instance.
The Qiskit-Braket provider is pre-installed on Amazon Braket notebook instances, so you can start
using it right away without any additional installation steps.

1. `Enable Amazon Braket <https://docs.aws.amazon.com/braket/latest/developerguide/braket-enable-overview.html>`_.
2. `Create an Amazon Braket notebook instance <https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-create-notebook.html>`_.

***********************************
Getting started in your environment
***********************************

You can install the Qiskit-Braket provider in your own environment after enabling Amazon Braket and
configuring the AWS SDK for Python:

1. `Enable Amazon Braket <https://docs.aws.amazon.com/braket/latest/developerguide/braket-enable-overview.html>`_.
2. Configure the AWS SDK for Python (Boto3) using the `Quickstart <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html>`_.
3. Install the Qiskit-Braket provider (see `Installing the Qiskit-Braket provider`_ below).

**Note:** Make sure that your AWS region is set to one supported by Amazon Braket. You can check this
in your AWS configuration file, which is located by default at ``~/.aws/config``.

****************************************
Installing the Qiskit-Braket provider
****************************************

The Qiskit-Braket provider can be installed with pip as follows:

.. code-block:: bash

    pip install qiskit-braket-provider

You can also install from source by cloning this repository and running a pip install command in the
root directory of the repository:

.. code-block:: bash

    git clone https://github.com/amazon-braket/qiskit-braket-provider.git
    cd qiskit-braket-provider
    pip install .

Check the version you have installed
====================================

You can view the version of ``qiskit-braket-provider`` you have installed by using the following
command:

.. code-block:: bash

    pip show qiskit-braket-provider

You can also check your version from within Python:

.. code-block:: python

    >>> import qiskit_braket_provider
    >>> qiskit_braket_provider.__version__

Updating the Qiskit-Braket provider
===================================

You can update your installed version by using the following command:

.. code-block:: bash

    pip install qiskit-braket-provider --upgrade --upgrade-strategy eager
