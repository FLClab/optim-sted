
import numpy
import time
import random

from pysted import base, utils, raster, bleach_funcs
from pysted.microscopes import DyMINMicroscope, RESCueMicroscope, DyMINRESCueMicroscope
from matplotlib import pyplot

import sys
sys.path.insert(0, "..")
from src.tools import datamap_generator
from src.experiment import Experiment
from src import defaults

from banditopt import objectives

resolution = objectives.Resolution(pixelsize=20e-9)

numpy.random.seed(42)
random.seed(42)

if __name__ == "__main__":

    START = time.time()

    delta = 1
    num_mol = 250
    # num_mol = 30
    molecules_disposition = numpy.zeros((50, 50))
    # molecules_disposition[
    #     molecules_disposition.shape[0]//2 - delta : molecules_disposition.shape[0]//2+delta+1,
    #     molecules_disposition.shape[1]//2 - delta : molecules_disposition.shape[1]//2+delta+1] = num_mol
    for j in range(1,4):
        for i in range(1,4):
            molecules_disposition[
                j * molecules_disposition.shape[0]//4 - delta : j * molecules_disposition.shape[0]//4 + delta + 1,
                i * molecules_disposition.shape[1]//4 - delta : i * molecules_disposition.shape[1]//4 + delta + 1] = num_mol


    molecules_disposition, positions = datamap_generator(
        shape = (50, 50),
        sources = 10,
        molecules = num_mol,
        shape_sources = (3, 3),
        random_state = None
    )

    action_spaces = defaults.action_spaces

    print("Setting up the microscope ...")
    pixelsize = 20e-9
    bleach = True
    p_ex = 0.02 * (action_spaces["p_ex"]["high"] - action_spaces["p_ex"]["low"]) + action_spaces["p_ex"]["low"]
    p_ex_array = numpy.ones(molecules_disposition.shape) * p_ex
    p_sted = 0.1 * (action_spaces["p_sted"]["high"] - action_spaces["p_sted"]["low"]) + action_spaces["p_sted"]["low"]
    p_sted_array = numpy.ones(molecules_disposition.shape) * p_sted
    pdt = 0. * (action_spaces["pdt"]["high"] - action_spaces["pdt"]["low"]) + action_spaces["pdt"]["low"]
    pdt = 100.0e-6
    pdt_array = numpy.ones(molecules_disposition.shape) * pdt
    roi = 'max'

    print(p_ex, p_sted, pdt)

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(**defaults.LASER_EX)
    laser_sted = base.DonutBeam(**defaults.LASER_STED)
    detector = base.Detector(**defaults.DETECTOR)
    objective = base.Objective(**defaults.OBJECTIVE)
    fluo = base.Fluorescence(**defaults.FLUO)
    datamap = base.Datamap(molecules_disposition, pixelsize)

    opts = {
        "scale_power" : [0., 0.25, 1.],
        "decision_time" : [10.0e-6, 10.0e-6, -1],
        "threshold_count" : [8, 8, 0]
    }
    microscope = DyMINMicroscope(laser_ex, laser_sted, detector, objective, fluo, opts=opts)
    opts = {
        "scale_power" : [0., 0.25, 1.],
        "decision_time" : [10.0e-6, 10.0e-6, 10.0e-6],
        "threshold_count" : [8, 8, 3]
    }
    # microscope = DyMINRESCueMicroscope(laser_ex, laser_sted, detector, objective, fluo, opts=opts)
    # opts = {
    #     "lower_threshold" : [2, -1.],
    #     "upper_threshold" : [4, -1.],
    #     "decision_time" : [6.0e-6, -1.]
    # }
    opts = {
        "lower_threshold" : [2, -1.],
        "upper_threshold" : [4, -1.],
        "decision_time" : [10.0e-6, -1.]
    }
    # microscope = RESCueMicroscope(laser_ex, laser_sted, detector, objective, fluo, opts=opts)
    # microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo)
    # microscope.opts = {"decision_time" : [0., 0.]}
    start = time.time()
    i_ex, i_sted, _ = microscope.cache(datamap.pixelsize, save_cache=True)

    datamap.set_roi(i_ex, roi)
    print("Setup done...")

    time_start = time.time()
    # acquisition, bleached, scaled_power = microscope.get_signal_and_bleach(datamap, datamap.pixelsize, pdt, p_ex, p_sted,
    #                                                                     bleach=bleach, update=False, seed=42)

    experiment = Experiment()
    experiment.add("STED", microscope, datamap, params={"pdt":pdt, "p_ex":p_ex, "p_sted":p_sted})
    histories = experiment.acquire_all(1, bleach=bleach, processes=1)
    history = histories["STED"]

    acquisition, bleached, scaled_power = history["acquisition"], history["bleached"], history["other"]

    print("DyMIN power", (numpy.sum(scaled_power[scaled_power == 1] * p_sted * pdt) + \
                         numpy.sum(scaled_power[scaled_power == 0.1] * p_sted * microscope.opts["decision_time"][1]) + \
                         numpy.sum(scaled_power[scaled_power == 0] * p_sted * microscope.opts["decision_time"][0])) * 1e+3)
    print("DyMINRESCue power", (numpy.sum(scaled_power[scaled_power == 1] * p_sted * pdt) + \
                         numpy.sum(scaled_power[scaled_power == 0.1] * p_sted * microscope.opts["decision_time"][1]) + \
                         numpy.sum(scaled_power[scaled_power == 0] * p_sted * microscope.opts["decision_time"][0]) + \
                         numpy.sum(scaled_power == 3) * p_sted * (pdt + microscope.opts["decision_time"][-1]) + \
                         numpy.sum(scaled_power == 2) * p_sted * microscope.opts["decision_time"][-1] + \
                         numpy.sum(scaled_power == 4) * p_sted * microscope.opts["decision_time"][-1]) * 1e+3)
    print("RESCue power", (numpy.sum(scaled_power == 1) * p_sted * pdt + \
                          numpy.sum(scaled_power == 0) * p_sted * microscope.opts["decision_time"][0] + \
                          numpy.sum(scaled_power == 2) * p_sted * microscope.opts["decision_time"][0]) * 1e+3)
    print("STED power", numpy.sum(pdt * p_sted * scaled_power.size) * 1e+3)

    print(f"ran in {time.time() - time_start} s")

    obj = resolution.evaluate(acquisition, None, None, None, None)
    print("Resolution", obj)

    fig, axes = pyplot.subplots(1, 4, figsize=(10,3), sharey=True, sharex=True)

    axes[0].imshow(datamap.whole_datamap[datamap.roi])
    axes[0].set_title(f"Datamap roi")

    axes[1].imshow(bleached[-1], vmin=0, vmax=datamap.whole_datamap[datamap.roi].max())
    axes[1].set_title(f"Bleached datamap")

    axes[2].imshow(acquisition[-1])
    axes[2].set_title(f"Acquired signal (photons)")

    axes[3].imshow(scaled_power[-1])
    axes[3].set_title(f"Scaled power")

    print("Average molecules before : ", datamap.whole_datamap[datamap.roi][molecules_disposition != 0].mean(axis=-1))
    print("Average molecules left : ", bleached[-1][molecules_disposition != 0].mean(axis=-1))
    print("Bleaching : ", [((datamap.whole_datamap[datamap.roi][molecules_disposition != 0] - bleached[i][molecules_disposition != 0]) / datamap.whole_datamap[datamap.roi][molecules_disposition != 0]).mean() for i in range(len(bleached))])

    print("Total run time : {}".format(time.time() - START))

    pyplot.show(block=True)
