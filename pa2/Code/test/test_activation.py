import unittest
import numpy as np
from Answer import ReLU, LeakyReLU, ELU
from utils import rel_error


class TestReLU(unittest.TestCase):
    def test_relu_1_forward(self):
        print('\n==================================')
        print('          Test ReLU forward       ')
        print('==================================')
        x = np.linspace(-0.7, 0.5, num=20).reshape(5, 4)
        relu = ReLU()
        out = relu.forward(x)
        correct_out = np.array([[0., 0., 0., 0.],
                                [0., 0., 0., 0.],
                                [0., 0., 0., 0.],
                                [0.05789474, 0.12105263, 0.18421053, 0.24736842],
                                [0.31052632, 0.37368421, 0.43684211, 0.5]])
        e = rel_error(correct_out, out)
        print('Relative difference:', e)
        self.assertTrue(e <= 5e-07)

    def test_relu_2_backward(self):
        print('\n==================================')
        print('          Test ReLU backward      ')
        print('==================================')
        np.random.seed(123)
        relu = ReLU()
        x = np.random.randn(7, 7)
        d_prev = np.random.randn(*x.shape)
        out = relu.forward(x)
        dx = relu.backward(d_prev, 0.0)
        correct_dx = [[0.        , -1.29408532, -1.03878821,  0.        ,  0.        ,  0.02968323, 0.        ],
                      [0.        ,  1.75488618,  0.        ,  0.        ,  0.        ,  0.79486267, 0.        ],
                      [0.        ,  0.        ,  0.80723653,  0.04549008, -0.23309206, -1.19830114, 0.19952407],
                      [0.46843912,  0.        ,  1.16220405,  0.        ,  0.        ,  1.03972709, 0.        ],
                      [0.        ,  0.        ,  0.        ,  0.        ,  0.        ,  0.        , 0.80730819],
                      [0.        , -1.0859024 , -0.73246199,  0.        ,  2.08711336,  0.        , 0.        ],
                      [0.        ,  0.18103513,  1.17786194,  0.        ,  1.03111446, -1.08456791, -1.36347154]]
        e = rel_error(correct_dx, dx)
        print('dX relative difference:', e)
        self.assertTrue(e <= 5e-08)

    def runTest(self):
        self.test_relu_1_forward()
        self.test_relu_2_backward()

class TestLeakyReLU(unittest.TestCase):
    def test_leakyrelu_1_forward(self):
        print('\n==================================')
        print('      Test Leaky ReLU forward       ')
        print('==================================')
        x = np.linspace(-0.7, 0.5, num=20).reshape(5, 4)
        leakyrelu = LeakyReLU()
        out = leakyrelu.forward(x)
        correct_out = np.array([[-0.07      , -0.06368421, -0.05736842, -0.05105263],
                                [-0.04473684, -0.03842105, -0.03210526, -0.02578947],
                                [-0.01947368, -0.01315789, -0.00684211, -0.00052632],
                                [ 0.05789474,  0.12105263,  0.18421053,  0.24736842],
                                [ 0.31052632,  0.37368421,  0.43684211,  0.5       ]])
        e = rel_error(correct_out, out)
        print('Relative difference:', e)
        self.assertTrue(e <= 5e-06)

    def test_leakyrelu_2_backward(self):
        print('\n==================================')
        print('      Test Leaky ReLU backward      ')
        print('==================================')
        np.random.seed(123)
        leakyrelu = LeakyReLU()
        x = np.random.randn(7, 7)
        d_prev = np.random.randn(*x.shape)
        out = leakyrelu.forward(x)
        dx = leakyrelu.backward(d_prev, 0.0)
        correct_dx = [[ 0.22381433, -1.29408532, -1.03878821,  0.17437122, -0.07980627,  0.02968323,   0.1069316 ],
                      [ 0.08907064,  1.75488618,  0.14956441,  0.10693927, -0.07727087,  0.79486267,   0.0314272 ],
                      [-0.13262655,  0.1417299,   0.80723653,  0.04549008, -0.23309206, -1.19830114,   0.19952407],
                      [ 0.46843912, -0.0831155,   1.16220405, -0.1097203,  -0.21231004,  1.03972709,  -0.0403366 ],
                      [-0.01260296, -0.08375167, -0.16059628,  0.12552374, -0.0688869,   0.16609525,   0.80730819],
                      [-0.03147581, -1.0859024,  -0.73246199, -0.12125231,  2.08711336,  0.01644412,   0.11502055],
                      [-0.1267352,   0.18103513,  1.17786194, -0.03350108,  1.03111446, -1.08456791,  -1.36347154]]
        e = rel_error(correct_dx, dx)
        print('dX relative difference:', e)
        self.assertTrue(e <= 5e-07)

    def runTest(self):
        self.test_leakyrelu_1_forward()
        self.test_leakyrelu_2_backward()


class TestELU(unittest.TestCase):
    def test_elu_1_forward(self):
        print('\n==================================')
        print('          Test ELU forward       ')
        print('==================================')
        x = np.linspace(-0.7, 0.5, num=20).reshape(5, 4)
        elu = ELU()
        out = elu.forward(x)
        correct_out = np.array([[-0.5034147 , -0.47103981, -0.43655424, -0.39982039],
                                [-0.36069167, -0.31901195, -0.27461493, -0.22732344],
                                [-0.17694878, -0.12328994, -0.06613282, -0.00524933],
                                [ 0.05789474,  0.12105263,  0.18421053,  0.24736842],
                                [ 0.31052632,  0.37368421,  0.43684211,  0.5       ]])
        e = rel_error(correct_out, out)
        print('Relative difference:', e)
        self.assertTrue(e <= 5e-07)

    def test_elu_2_backward(self):
        print('\n==================================')
        print('          Test ELU backward      ')
        print('==================================')
        np.random.seed(123)
        elu = ELU()
        x = np.random.randn(7, 7)
        d_prev = np.random.randn(*x.shape)
        out = elu.forward(x)
        dx = elu.backward(d_prev, 0.0)
        correct_dx = np.array([[ 0.75579587, -1.29408532, -1.03878821,  0.38663337, -0.44745991,  0.02968323,   0.09445232],
                               [ 0.58004299,  1.75488618,  0.62864823,  0.54237629, -0.70288492,  0.79486267,   0.1658953 ],
                               [-0.85076878,  0.91796291,  0.80723653,  0.04549008, -0.23309206, -1.19830114,   0.19952407],
                               [ 0.46843912, -0.32602772,  1.16220405, -0.3131364,  -1.12201474,  1.03972709,  -0.09665653],
                               [-0.10955733, -0.35378392, -1.24371648,  0.07643883, -0.11715736,  0.82490586,   0.80730819],
                               [-0.26458667, -1.0859024,  -0.73246199, -0.50316712,  2.08711336,  0.07349275,   0.20438929],
                               [-0.85729768,  0.18103513,  1.17786194, -0.33107077,  1.03111446, -1.08456791,  -1.36347154]])
        e = rel_error(correct_dx, dx)
        print('dX relative difference:', e)
        self.assertTrue(e <= 5e-08)

    def runTest(self):
        self.test_elu_1_forward()
        self.test_elu_2_backward()