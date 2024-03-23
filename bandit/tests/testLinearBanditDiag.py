
import numpy
import random
import sys
import torch
sys.path.insert(0, "..")
import matplotlib
matplotlib.use("macosx")

from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot
from scipy import stats

from banditopt import algorithms

# numpy.random.seed(42)
# random.seed(42)

class Sampler:
    def __init__(self, degree=3):
        self.degree = degree
        self._min, self._max = -1, 1

    def restart(self):
        X = numpy.zeros((1, 1))
        dim = PolynomialFeatures(self.degree).fit_transform(X).size
        self.weights = numpy.random.normal(0, 1.0, size=(dim, 1))

        # self.weights = numpy.array([
        #     [-0.90826427],
        #     [-0.59085011],
        #     [ 0.750179  ],
        #     [ 0.24555921],
        #     [ 0.77152652],
        #     [ 0.94150997]])
        self.weights = numpy.array([
            [-0.92268741],
            [ 1.71766238],
            [ 0.23203444],
            [-2.42728556],
            [ 0.57875611],
            [ 0.14331964]           
        ])

    def sample(self, num, sigma=0.05, X=None):
        if isinstance(X, type(None)):
            X = numpy.random.rand(num) * (self._max - self._min) + self._min
            X = X[:, numpy.newaxis]
        X_transformed = PolynomialFeatures(self.degree).fit_transform(X)
        noise = numpy.random.normal(0, sigma, size=(len(X_transformed), 1))
        return X, numpy.dot(X_transformed, self.weights) + noise

SIGMA = 0.25

bandit = algorithms.LinearBanditDiag(
    1, param_space_bounds=[(-1, 1)],
    _lambda=0.25, nu=0.5,
    # _lambda=10., nu=0.1,
    learning_rate=1e-2, style="TS"
)
sampler = Sampler(degree=5)
sampler.restart()
print(sampler.weights)

X = numpy.linspace(-1, 1)[:, numpy.newaxis]

sample_X, sample_y = [], []
for n in range(50):

    # bandit._lambda = (n / 200) ** 2 * (1000 - 0.1) + 0.1
    # bandit.U = bandit._lambda * torch.ones((bandit.total_param,))
    # sx, sy = sampler.sample(n - len(sample_X), sigma=0.1)
    if len(sample_X)>0:
        xsample = bandit.sample(X)
        sx = numpy.argmin(xsample, axis=0)
        sx = X[sx]
        sx, sy = sampler.sample(n - len(sample_X), sigma=SIGMA, X=sx)
    else:
        sx, sy = sampler.sample(1, sigma=SIGMA)

    sample_X.extend(sx)
    sample_y.extend(sy)
    bandit.update(
        numpy.array(sample_X), numpy.array(sample_y)
    )

    if len(sample_X) % 5 == 0:
        fig, ax = pyplot.subplots()
        X = numpy.linspace(-1, 1)[:, numpy.newaxis]
        X_transformed = PolynomialFeatures(sampler.degree).fit_transform(X)
        y = numpy.dot(X_transformed, sampler.weights)

        y = bandit.scaler.transform(y)

        ax.plot(X.ravel(), y.ravel())
        s_y = bandit.scaler.transform(numpy.array(sample_y))
        # print(s_y.mean(), s_y.std())
        ax.scatter(sample_X, s_y)

        kernel = stats.gaussian_kde(numpy.array(sample_X).T)
        kde = kernel(X.T)
        print(kde.min())

        mean, std = bandit.get_mean_std(X)

        # std = std.ravel() * (1 / (kde.ravel()))
        std = std.ravel()
        mean = bandit.scaler.transform(mean)

        ax.plot(X.ravel(), mean.ravel())
        ax.fill_between(X.ravel(), mean.ravel() - std.ravel(), mean.ravel() + std.ravel(), alpha=0.3, color="tab:orange")

        ax.set_ylim(-3, 3)

        for _ in range(5):
            sample = bandit.sample(X)
            sample = bandit.scaler.transform(sample)
            ax.plot(X.ravel(), sample.ravel(), color="k", alpha=0.3)
        ax.set_title(n)
        pyplot.show(block=True)
