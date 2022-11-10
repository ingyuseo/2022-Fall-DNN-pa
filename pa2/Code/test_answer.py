import unittest

from test.test_activation import TestReLU, TestLeakyReLU, TestELU
from test.test_softmax_layer import TestSoftmaxLayer
from test.test_fc_layer import TestFCLayer

"""

Here, we provide codes to check your implementation.
Each 'Test*' class check whether your implementation yields correct answers within acceptable error range.
You can check part of your implementation by only giving test instances you want in variable 'test_cases'.
By default, the codes will check all 6 part of your answer.

"""
test_cases = [TestReLU(), TestLeakyReLU(), TestELU(), TestSoftmaxLayer(), TestFCLayer()]

if __name__ == '__main__':
    suite = unittest.TestSuite(test_cases)
    unittest.TextTestRunner().run(suite)
