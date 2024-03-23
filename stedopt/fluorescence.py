
import numpy
import random

from scipy import optimize
from skimage import filters

from stedopt import defaults
from stedopt.create import create_microscope

class FluorescenceOptimizer():
    """
    Optimizes the parameters of a fluorophore.
    """
    FACTORS = {
        "clusters" : 2.25, # CaMKII & PSD95
        "actin" : 3.0,
        "tubulin" : 3.75
    }

    def __init__(self, microscope=None, sample="clusters", iterations=10, pixelsize=20e-9):
        """
        Instantiates the `FluorescenceOptimizer`

        :param microscope: A `pysted.base.Microscope` object
        :param sample: A `str` of the sample type that is being optimized
        :param iterations: An `int` of the number of iterations to perform
        :param pixelsize: A `float` of the pixel size
        """
        self.microscope = microscope
        if isinstance(self.microscope, type(None)):
            self.microscope, _, _ = create_microscope()
        self.iterations = iterations
        self.pixelsize = pixelsize

        assert sample in ["clusters", "actin", "tubulin"]
        self.correction_factor = self.FACTORS[sample]
        self.scale_factor = 40.

        # This seems to be optimal for a starting point
        self.microscope.fluo.sigma_abs = defaults.FLUO["sigma_abs"]
        self.microscope.fluo.k1 = defaults.FLUO["k1"]
        self.microscope.fluo.b = defaults.FLUO["b"]

    def default_parameters(self):
        """
        Returns the default parameters
        """
        self.microscope.fluo.sigma_abs = defaults.FLUO["sigma_abs"]
        self.microscope.fluo.k1 = defaults.FLUO["k1"]
        self.microscope.fluo.b = defaults.FLUO["b"]
        return defaults.FLUO["k1"], defaults.FLUO["b"], defaults.FLUO["sigma_abs"]

    def aggregate(self, criterions, **kwargs):
        """
        Aggregates the returned values

        :param criterions: A `dict` of criterions
        """
        out = {}
        if "bleach" in criterions:
            out["bleach"] = {
                "k1" : kwargs.get("k1") * 1e-15,
                "b" : kwargs.get("b")
            }
        if "signal" in criterions:
            sigma_abs = self.microscope.fluo.sigma_abs
            out["signal"] = {
                "sigma_abs" : {
                    int(self.microscope.excitation.lambda_ * 1e9) : kwargs.get("sigma_abs") * 1e-20,
                    int(self.microscope.sted.lambda_ * 1e9) : sigma_abs[int(self.microscope.sted.lambda_ * 1e9)]
                }
            }
        return out

    def optimize(self, criterions):
        """
        Optimizes the fluorescene parameters given the input criterions.

        The optimization of the parameters is done
        sequentially has this seems to produce better results on the datamaps
        that were tested. However, a multi-objective approach (e.g. NSGA-II)
        could be better suited to find the Pareto choices.

        :param criterions: A `dict` of criterions

        :returns : A `dict` of the optimized parameters

        :example :

        criterions = {
            "bleach" : {
                "p_ex" : <VALUE>,
                "p_sted" : <VALUE>,
                "pdt" : <VALUE>,
                "target" : <VALUE>
            },
            "signal" : {
                "p_ex" : <VALUE>,
                "p_sted" : <VALUE>,
                "pdt" : <VALUE>,
                "target" : <VALUE>
            },
        }
        params = optimizer.optimize(criterions)
        >>> params
        {
            "bleach" : {
                "k1" : <VALUE>,
                "b" : <VALUE>
            },
            "signal" : {
                "sigma_abs" : {
                    635 : <VALUE>,
                    750 : <VALUE>
                }
            }
        }
        """
        k1, b, sigma_abs = self.default_parameters()
        sigma_abs = sigma_abs[int(self.microscope.excitation.lambda_ * 1e9)]
        for _ in range(self.iterations):
            params = criterions.get("bleach", None)
            if params:
                # Optimize bleaching constants
                res = optimize.minimize(
                    self.optimize_bleach, x0=[k1, b],
                    args=(params["p_ex"], params["p_sted"], params["pdt"], params["target"]),
                    options={"eps" : 0.01, "maxiter": 100}, tol=1e-3,
                    bounds = [(0., numpy.inf), (0., 5.0)]
                )
                k1, b = res.x

            # Optimize signal constant
            params = criterions.get("signal", None)
            if params:
                res = optimize.minimize(
                    self.optimize_sigma_abs, x0=[sigma_abs],
                    args=(params["p_ex"], params["p_sted"], params["pdt"], params["target"]),
                    options={"eps" : 0.01, "maxiter": 100}, tol=1e-3,
                    bounds = [(0., numpy.inf)]
                )
                sigma_abs = res.x.item()
        return self.aggregate(criterions, k1=k1, b=b, sigma_abs=sigma_abs)

    def kb_map_to_im_bleach(self, kb_map, dwelltime, linestep):
        """
        Bleaching estimate for an infinite number of fluorophores
        kb_map being the map of k_bleach convolving each pixel
        """
        return 1 - numpy.exp((-kb_map * dwelltime * linestep).sum())

    def expected_bleach(self, p_ex, p_sted, pdt):
        """
        Calculates the expected confocal signal given some parameters

        :param p_ex: A `float` of the excitation power
        :param p_sted: A `float` of the STED power
        :param pdt: A `float` of the pixel dwelltime

        :returns : A `int` of the expected number of photons
        """
        __i_ex, __i_sted, psf_det = self.microscope.cache(self.pixelsize)

        i_ex = __i_ex * p_ex #the time averaged excitation intensity
        i_sted = __i_sted * p_sted #the instant sted intensity (instant p_sted = p_sted/(self.sted.tau * self.sted.rate))

        lambda_ex, lambda_sted = self.microscope.excitation.lambda_, self.microscope.sted.lambda_
        tau_sted = self.microscope.sted.tau
        tau_rep = 1 / self.microscope.sted.rate
        phi_ex =  self.microscope.fluo.get_photons(i_ex, lambda_=lambda_ex)
        phi_sted = self.microscope.fluo.get_photons(i_sted, lambda_=lambda_sted)

        kb_map = self.microscope.fluo.get_k_bleach(
            lambda_ex, lambda_sted, phi_ex, phi_sted*tau_sted/tau_rep, tau_sted,
            tau_rep, pdt
        )
        bleach = self.kb_map_to_im_bleach(kb_map, pdt, 1)
        return bleach

    def expected_confocal_signal(self, p_ex, p_sted, pdt):
        """
        Calculates the expected confocal signal given some parameters

        :param p_ex: A `float` of the excitation power
        :param p_sted: A `float` of the STED power
        :param pdt: A `float` of the pixel dwelltime

        :returns : A `int` of the expected number of photons
        """
        photons_mean = []

        effective = self.microscope.get_effective(self.pixelsize, p_ex, p_sted)
        datamap = numpy.zeros_like(effective)
        cy, cx = (s // 2 for s in datamap.shape)
        datamap[cy, cx] = 1
        datamap = filters.gaussian(datamap, sigma=self.correction_factor)
        datamap = datamap / datamap.max() * self.scale_factor

        intensity = numpy.sum(effective * datamap)
        photons = self.microscope.fluo.get_photons(intensity)

        # The calculation is repeated since there is randomness
        for _ in range(25):
            p = self.microscope.detector.get_signal(photons, pdt, self.microscope.sted.rate)
            photons_mean.append(p)
        photons = numpy.mean(photons_mean)
        return photons

    def optimize_bleach(self, x, p_ex, p_sted, pdt, target):
        """
        Method used by `scipy.optimize.minimize` to optimize the
        photobleaching
        """
        k1, b = x
        self.microscope.fluo.k1 = k1 * 1e-15
        self.microscope.fluo.b = b
        bleach = self.expected_bleach(p_ex, p_sted, pdt)
        error = (target - bleach) ** 2
        return error

    def optimize_sigma_abs(self, sigma_abs, p_ex, p_sted, pdt, target):
        """
        Method used by `scipy.optimize.minimize` to optimize the
        signal.

        Note. The error signal is normalized by the target to obtain
        reasonable error value during the optimization.
        """
        self.microscope.fluo.sigma_abs = {
            int(self.microscope.excitation.lambda_ * 1e9) : sigma_abs * 1e-20,
            int(self.microscope.sted.lambda_ * 1e9): self.microscope.fluo.sigma_abs[int(self.microscope.sted.lambda_ * 1e9)],
        }

        signal = self.expected_confocal_signal(p_ex, 0., pdt)
        error = ((target - signal) / target) ** 2

        return error

class Criterion:
    """
    Implements a base `Criterion` that can be used to optimize the parameters of
    fluorescence
    """
    def __init__(self, criterions):
        """
        Instantiates the `Criterion`

        :param criterions: A `dict` of criterions
        """
        self.criterions = criterions

    def get(self, item, default=None):
        if item not in self.criterions:
            return default
        return self.criterions[item]

    def __contains__(self, item):
        return item in self.criterions

    def __str__(self):
        return str(self.criterions)

class ChoiceCriterion(Criterion):
    """
    Implements a `RandomCriterion` that can be used to optimize the parameters of
    fluorescence. This criterion allows to randomly sample from a range of
    parameters.
    """
    def __init__(self, criterions):
        super().__init__(criterions)
        self.criterions = {
            key : self.sample(value) for key, value in self.criterions.items()
        }

    def sample(self, criterion):
        """
        Samples from the given criterions

        :param criterion: A `dict` parameters and values to sample from
        """
        criterion = criterion.copy()
        for key, values in criterion.items():
            if isinstance(values, (tuple, list)):
                possible = numpy.linspace(*values, 5)
                criterion[key] = random.choice(possible).item()
            else:
                criterion[key] = values
        return criterion

class UniformCriterion(Criterion):
    """
    Implements a `RandomCriterion` that can be used to optimize the parameters of
    fluorescence. This criterion allows to randomly sample from a range of
    parameters.
    """
    def __init__(self, criterions):
        super().__init__(criterions)
        self.criterions = {
            key : self.sample(value) for key, value in self.criterions.items()
        }

    def sample(self, criterion):
        """
        Samples from the given criterions

        :param criterion: A `dict` parameters and values to sample from
        """
        criterion = criterion.copy()
        for key, values in criterion.items():
            if isinstance(values, (tuple, list)):
                criterion[key] = random.uniform(*values)
            else:
                criterion[key] = values
        return criterion


if __name__ == "__main__":

    criterions = {
        "bleach" : {
            "p_ex" : [1.0e-6, 10.0e-6],
            "p_sted" : [100e-3, 200e-3],
            "pdt" : [1.0e-6, 10.0e-6],
            "target" : [0.1, 0.9]
        },
        "signal" : { # Avoids breaking the microscope
            "p_ex" : 10.0e-6,
            "p_sted" : 0.,
            "pdt" : 10.0e-6,
            "target" : 200.
        },
    }

    criterions = {
        "bleach" : {
            "p_ex" : 5.0e-6,
            "p_sted" : [100e-3, 200e-3],
            "pdt" : 10e-6,
            "target" : [0.1, 0.9]
        },
        "signal" : { # Avoids breaking the microscope
            "p_ex" : 10.0e-6,
            "p_sted" : 0.,
            "pdt" : [10.0e-6, 50e-6],
            "target" : 200.
        },
    }

    import random
    import json, os
    from tqdm.auto import trange
    import time

    random.seed(42)
    for sample in ["clusters", "actin", "tubulin"]:

        routines = []
        for _ in trange(3):
            fluo = defaults.FLUO.copy()

            criterion = UniformCriterion(criterions)
            # print(criterion)
            optimizer = FluorescenceOptimizer(sample=sample, iterations=10)
            start = time.time()
            out = optimizer.optimize(criterion)
            print(time.time() - start)

            for criterion, values in out.items():
                for key, value in values.items():
                    fluo[key] = value

            routines.append({
                "fluo" : fluo
            })

        # json.dump(routines, open(os.path.join(f"./routines/generated-routines-{sample}.json"), "w"), indent=4, sort_keys=True)
