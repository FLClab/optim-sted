
import numpy
import random
import sys
sys.path.insert(0, "..")

from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot

import algorithms

# numpy.random.seed(42)
# random.seed(42)

class Sampler:
    def __init__(self, degree=3):
        self.degree = degree
        self._min, self._max = -1, 1

    def restart(self):
        X = numpy.zeros((1, 1))
        dim = PolynomialFeatures(self.degree).fit_transform(X).size
        self.weights = numpy.random.normal(0, 2.0, size=(dim, 1))

    def sample(self, num, sigma=0.05):
        X = numpy.random.rand(num) * (self._max - self._min) + self._min
        X = X[:, numpy.newaxis]
        X_transformed = PolynomialFeatures(self.degree).fit_transform(X)
        noise = numpy.random.normal(0, sigma, size=(len(X_transformed), 1))
        return X, numpy.dot(X_transformed, self.weights) + noise

bandit = algorithms.sklearn_GP(4/3, 0.001, 0.3, [(-1, 1)])
sampler = Sampler(degree=3)
sampler.restart()

sample_X, sample_y = [], []
for n in [5, 10, 15, 25, 50, 100, 150, 200]:
    sx, sy = sampler.sample(n - len(sample_X), sigma=0.5)
    sample_X.extend(sx)
    sample_y.extend(sy)
    bandit.update(
        numpy.array(sample_X), numpy.array(sample_y)
    )

    fig, ax = pyplot.subplots()
    X = numpy.linspace(-1, 1)[:, numpy.newaxis]
    X_transformed = PolynomialFeatures(sampler.degree).fit_transform(X)
    y = numpy.dot(X_transformed, sampler.weights)
    ax.plot(X.ravel(), y.ravel())
    ax.scatter(sample_X, sample_y)

    mean, std = bandit.get_mean_std(X)
    ax.plot(X.ravel(), mean.ravel())
    ax.fill_between(X.ravel(), mean.ravel() - std.ravel(), mean.ravel() + std.ravel(), alpha=0.3, color="tab:orange")

    ax.set_ylim(-5, 5)

    for _ in range(25):
        sample = bandit.sample(X)
        ax.plot(X.ravel(), sample.ravel(), color="k", alpha=0.3)

    pyplot.show()
