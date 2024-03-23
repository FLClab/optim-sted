
'''
Module used to create the microscope objects used in the simulated experiments
'''

import numpy
import pickle

from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

# from . import tools

from pysted import base, utils
from pysted.microscopes import DyMINMicroscope, RESCueMicroscope

from . import defaults

def create_dymin_microscope(**kwargs):
    '''
    Creates a DyMIN microscope
    '''
    # Creates a default datamap
    delta = 1
    num_mol = 2
    ncols = 4
    nrows = 4
    molecules_disposition = numpy.zeros((50, 50))
    for j in range(1,nrows):
        for i in range(1,ncols):
            molecules_disposition[
                j * molecules_disposition.shape[0]//nrows - delta : j * molecules_disposition.shape[0]//nrows + delta + 1,
                i * molecules_disposition.shape[1]//ncols - delta : i * molecules_disposition.shape[1]//ncols + delta + 1] = num_mol

    # Extracts params
    laser_ex_params = kwargs.get("laser_ex", defaults.LASER_EX)
    laser_sted_params = kwargs.get("laser_sted", defaults.LASER_STED)
    detector_params = kwargs.get("detector", defaults.DETECTOR)
    objective_params = kwargs.get("objective", defaults.OBJECTIVE)
    fluo_params = kwargs.get("fluo", defaults.FLUO)
    datamap_params = kwargs.get("datamap", {
        "whole_datamap" : molecules_disposition,
        "datamap_pixelsize" : 20e-9
    })
    microscope_params = kwargs.get("microscope", {
        "scale_power" : [0., 0.25, 1.],
        "decision_time" : [10e-6, 10e-6, -1],
        "threshold_count" : [10, 8, 0]
    })
    imaging_params = kwargs.get("imaging", {
        "p_sted" : defaults.P_STED,
        "p_ex" : defaults.P_EX,
        "pdt" : defaults.PDT
    })

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(**laser_ex_params)
    laser_sted = base.DonutBeam(**laser_sted_params)
    detector = base.Detector(**detector_params)
    objective = base.Objective(**objective_params)
    fluo = base.Fluorescence(**fluo_params)
    datamap = base.Datamap(**datamap_params)

    microscope = DyMINMicroscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True, opts=microscope_params)
    i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)

    datamap.set_roi(i_ex, "max")

    return microscope, datamap, imaging_params

def create_rescue_microscope(**kwargs):
    '''
    Creates a RESCue microscope
    '''
    # Creates a default datamap
    delta = 1
    num_mol = 2
    ncols = 4
    nrows = 4
    molecules_disposition = numpy.zeros((50, 50))
    for j in range(1,nrows):
        for i in range(1,ncols):
            molecules_disposition[
                j * molecules_disposition.shape[0]//nrows - delta : j * molecules_disposition.shape[0]//nrows + delta + 1,
                i * molecules_disposition.shape[1]//ncols - delta : i * molecules_disposition.shape[1]//ncols + delta + 1] = num_mol

    # Extracts params
    laser_ex_params = kwargs.get("laser_ex", defaults.LASER_EX)
    laser_sted_params = kwargs.get("laser_sted", defaults.LASER_STED)
    detector_params = kwargs.get("detector", defaults.DETECTOR)
    objective_params = kwargs.get("objective", defaults.OBJECTIVE)
    fluo_params = kwargs.get("fluo", defaults.FLUO)
    datamap_params = kwargs.get("datamap", {
        "whole_datamap" : molecules_disposition,
        "datamap_pixelsize" : 20e-9
    })
    microscope_params = kwargs.get("microscope", {
        "lower_threshold" : [2, 1.],
        "upper_threshold" : [7, -1],
        "decision_time" : [25e-6, -1]
    })
    imaging_params = kwargs.get("imaging", {
        "p_sted" : defaults.P_STED,
        "p_ex" : defaults.P_EX,
        "pdt" : defaults.PDT
    })

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(**laser_ex_params)
    laser_sted = base.DonutBeam(**laser_sted_params)
    detector = base.Detector(**detector_params)
    objective = base.Objective(**objective_params)
    fluo = base.Fluorescence(**fluo_params)
    datamap = base.Datamap(**datamap_params)

    microscope = RESCueMicroscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True, opts=microscope_params)
    i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)

    datamap.set_roi(i_ex, "max")

    return microscope, datamap, imaging_params

def create_microscope(**kwargs):
    '''
    Creates a STED microscope.

    .. note:: 
        A confocal microscope can be created by passing an imaging option
        with a `"p_sted" = 0.`
    '''
    # Creates a default datamap
    delta = 1
    num_mol = 2
    molecules_disposition = numpy.zeros((50, 50))
    for j in range(1,4):
        for i in range(1,4):
            molecules_disposition[
                j * molecules_disposition.shape[0]//4 - delta : j * molecules_disposition.shape[0]//4 + delta + 1,
                i * molecules_disposition.shape[1]//4 - delta : i * molecules_disposition.shape[1]//4 + delta + 1] = num_mol

    # Extracts params
    laser_ex_params = kwargs.get("laser_ex", defaults.LASER_EX)
    laser_sted_params = kwargs.get("laser_sted", defaults.LASER_STED)
    detector_params = kwargs.get("detector", defaults.DETECTOR)
    objective_params = kwargs.get("objective", defaults.OBJECTIVE)
    fluo_params = kwargs.get("fluo", defaults.FLUO)
    datamap_params = kwargs.get("datamap", {
        "whole_datamap" : molecules_disposition,
        "datamap_pixelsize" : 20e-9
    })
    imaging_params = kwargs.get("imaging", {
        "p_sted" : defaults.P_STED,
        "p_ex" : defaults.P_EX,
        "pdt" : defaults.PDT
    })

    # Generating objects necessary for acquisition simulation
    laser_ex = base.GaussianBeam(**laser_ex_params)
    laser_sted = base.DonutBeam(**laser_sted_params)
    detector = base.Detector(**detector_params)
    objective = base.Objective(**objective_params)
    fluo = base.Fluorescence(**fluo_params)
    datamap = base.Datamap(**datamap_params)

    microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
    i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)

    datamap.set_roi(i_ex, "max")

    return microscope, datamap, imaging_params

class MicroscopeGenerator():
    def __init__(self):

        # Creates a default datamap
        delta = 1
        num_mol = 2
        self.molecules_disposition = numpy.zeros((50, 50))
        for j in range(1,4):
            for i in range(1,4):
                self.molecules_disposition[
                    j * self.molecules_disposition.shape[0]//4 - delta : j * self.molecules_disposition.shape[0]//4 + delta + 1,
                    i * self.molecules_disposition.shape[1]//4 - delta : i * self.molecules_disposition.shape[1]//4 + delta + 1] = num_mol

        # Extracts params
        self.laser_ex_params = kwargs.get("laser_ex", defaults.LASER_EX)
        self.laser_sted_params = kwargs.get("laser_sted", defaults.LASER_STED)
        self.detector_params = kwargs.get("detector", defaults.DETECTOR)
        self.objective_params = kwargs.get("objective", defaults.OBJECTIVE)
        self.fluo_params = kwargs.get("fluo", defaults.FLUO)
        self.pixelsize = 20e-9

    def generate(self, **kwargs):

        datamap_params = kwargs.get("datamap", {
            "whole_datamap" : molecules_disposition,
            "datamap_pixelsize" : self.pixelsize
        })
        imaging_params = kwargs.get("imaging", {
            "p_sted" : defaults.P_STED,
            "p_ex" : defaults.P_EX,
            "pdt" : defaults.PDT
        })

        # Generating objects necessary for acquisition simulation
        laser_ex = base.GaussianBeam(**self.laser_ex_params)
        laser_sted = base.DonutBeam(**self.laser_sted_params)
        detector = base.Detector(**self.detector_params)
        objective = base.Objective(**self.objective_params)
        fluo = base.Fluorescence(**self.fluo_params)
        datamap = base.Datamap(**datamap_params)

        microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
        i_ex, _, _ = microscope.cache(datamap.pixelsize, save_cache=True)

        datamap.set_roi(i_ex, "max")
        return microscope, datamap, imaging_params
