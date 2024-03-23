import os
import functools
import warnings
import time
import csv
import matplotlib; matplotlib.use("TkAgg")
from matplotlib import pyplot
import numpy
import skimage.io
#from skimage.external.tifffile import TiffWriter
import yaml
import microscope
import user
import utils
from datetime import date


configs = microscope.get_config("Getting RESCue/DyMIN configs")
with open("dymin-configs.yml", "w") as file:
    yaml.dump(configs.parameters(""), file)
