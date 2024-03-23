
import numpy
import random
import sys
sys.path.insert(0, "..")

from scipy import stats, special
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot
import matplotlib; matplotlib.use("MacOSX")
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
        self.weights = numpy.array([
            [-0.43213487],
            [-1.1544024 ],
            [ 1.63174786],
            [ 0.40412351],
            [-0.75440282],
            [ 0.93457183],
            [ 2.13826454],
        ])

    def sample(self, num, sigma=0.05, X=None):
        if isinstance(X, type(None)):
            X = numpy.random.rand(num) * (self._max - self._min) + self._min
            X = X[:, numpy.newaxis]
        X_transformed = PolynomialFeatures(self.degree).fit_transform(X)
        noise = numpy.random.normal(0, sigma, size=(len(X_transformed), 1))
        return X, numpy.dot(X_transformed, self.weights) + noise


# Noise parameter (alpha)
x = numpy.linspace(0, 5)
alpha_1, alpha_2 = 1.0e-6, 1.0e-6
alpha_1, alpha_2 = 0.1, 1.0
pdf = stats.gamma.pdf(x, a=alpha_1, scale=1/alpha_2)
# pdf = x ** (alpha_1 - 1) * numpy.exp(-1 * alpha_2 * x) * alpha_2 ** alpha_1 / special.gamma(alpha_1)

fig, ax = pyplot.subplots()
ax.plot(x, pdf)
ax.set_title("Noise parameter (alpha)")
pyplot.show(block=True)

bandit = algorithms.sklearn_BayesRidge(
    alpha_1 = alpha_1,
    alpha_2 = alpha_2,
    alpha_init = None,
    compute_score = True,
    degree = 3,
    fit_intercept = False,
    lambda_1 = 0.1,
    lambda_2 = 10.,
    lambda_init = None,
    tol= 1.0e-06,
    param_space_bounds=[(-1., 1.)]
)
sampler = Sampler(degree=6)
sampler.restart()

X = numpy.linspace(-1, 1)[:, numpy.newaxis]

sample_X, sample_y = [], []
for idx, n in enumerate(range(1, 25)):
    fig, axes = pyplot.subplots(1, 2, figsize=(10, 5))
    if idx > 0:
        xsample = bandit.sample(X)
        sx = numpy.argmin(xsample, axis=0)
        sx = X[sx]
        sx, sy = sampler.sample(n - len(sample_X), sigma=0.5, X=sx)
        axes[0].plot(X.ravel(), xsample.ravel(), color="red", zorder=3)
        axes[0].axvline(sx[0], color="red", zorder=3, linestyle="dashed")
    else:
        sx, sy = sampler.sample(n - len(sample_X), sigma=0.5)
    sample_X.extend(sx)
    sample_y.extend(sy)
    bandit.update(
        numpy.array(sample_X), numpy.array(sample_y)
    )

    rescaled_X = algorithms.rescale_X(numpy.array(sample_X), bandit.param_space_bounds)
    axes[1].scatter(
        rescaled_X,
        bandit.scaler.transform(numpy.array(sample_y))
    )
    rescaled_X_features = PolynomialFeatures(bandit.degree).fit_transform(X)
    prediction, prediction_std = bandit.predict(rescaled_X_features, return_std=True)
    axes[1].plot(
        algorithms.rescale_X(X, bandit.param_space_bounds).ravel(),
        prediction.ravel()
    )
    axes[1].fill_between(
        algorithms.rescale_X(X, bandit.param_space_bounds).ravel(),
        prediction.ravel() - prediction_std.ravel(),
        prediction.ravel() + prediction_std.ravel(),
        color="tab:blue", alpha=0.3
    )

    axes[0].set_title(n)
    X_transformed = PolynomialFeatures(sampler.degree).fit_transform(X)
    y = numpy.dot(X_transformed, sampler.weights)
    axes[0].plot(X.ravel(), y.ravel())
    axes[0].scatter(sample_X, sample_y)
    argmin = numpy.argmin(y.ravel())
    axes[0].scatter(X.ravel()[argmin], y.ravel()[argmin], s=100, color="gold", zorder=3)

    mean, std = bandit.get_mean_std(X)
    std = std * bandit.scaler.scale_
    print("alpha:", bandit.alpha_, "lambda:", bandit.lambda_)

    axes[0].plot(X.ravel(), mean.ravel())
    axes[0].fill_between(X.ravel(), mean.ravel() - std.ravel(), mean.ravel() + std.ravel(), alpha=0.3, color="tab:orange")

    delta = y.max() - y.min()
    axes[0].set_ylim(y.min() - delta, y.max() + delta)
    axes[1].set_ylim(-5, 5)
    axes[1].set_xlim(-1, 1)

    for _ in range(25):
        sample = bandit.sample(X)
        axes[0].plot(X.ravel(), sample.ravel(), color="k", alpha=0.3)

    pyplot.show(block=True)
