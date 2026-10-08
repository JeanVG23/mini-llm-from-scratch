import numpy as np

from tests.gradcheck import numerical_grad, relative_error


def test_numerical_grad_matches_a_known_derivative():
    x = np.array([[1.0, -2.0], [0.5, 3.0]])
    grad = numerical_grad(lambda: float((x**2).sum()), x)
    assert np.allclose(grad, 2 * x, atol=1e-6)


def test_numerical_grad_restores_the_parameter():
    x = np.array([1.0, 2.0, 3.0])
    before = x.copy()
    numerical_grad(lambda: float(x.sum()), x)
    assert np.array_equal(x, before)


def test_relative_error_detects_a_wrong_gradient():
    good = np.array([1.0, 2.0, 3.0])
    assert relative_error(good, good) == 0.0
    assert relative_error(good, good * 1.1) > 1e-2
