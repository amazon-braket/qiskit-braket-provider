"""Tests for Braket jobs"""

from unittest import TestCase

from qiskit_braket_provider import AmazonBraketTask, AWSBraketJob


class TestAmazonBraketTask(TestCase):
    """Tests Amazon Braket Task"""

    def test_deprecation_warning_on_init(self):
        """Test to check if a deprecation warning is raised when initializing AmazonBraketTask"""
        with self.assertWarns(DeprecationWarning):

            class SubAmazonBraketTask(AmazonBraketTask):
                """Subclass of AmazonBraketTask for testing"""


class TestAWSBraketJob(TestCase):
    """Tests Amazon Braket Job"""

    def test_deprecation_warning_on_init(self):
        """Test to check if a deprecation warning is raised when initializing AWSBraketJob"""
        with self.assertWarns(DeprecationWarning):

            class SubAwsBraketJob(AWSBraketJob):
                """Subclass of AWSBraketJob for testing"""
