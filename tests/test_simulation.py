from gaussian_processes_variational.simulation import LinearSimulator, Simulator, FriedMan1Simulator, RBFSimulator
from typing import Type


def model_test(simulator_class: Type[Simulator]):
    n = 20
    n_input = 10
    simulator = simulator_class(n)
    data = simulator.simulate(n_input)
    assert len(data.X_train) == n
    assert len(data.X_test) == n
    assert len(data.X_train[0]) == n_input
    assert len(data.X_test[0]) == n_input


def test_linear():
    model_test(LinearSimulator)


def test_friedman1():
    model_test(FriedMan1Simulator)


def test_rbf():
    model_test(RBFSimulator)
