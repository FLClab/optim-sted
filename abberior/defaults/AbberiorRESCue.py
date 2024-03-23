
from functools import partial

from abberior import microscope

EXC_ID = 3
STED_ID = 6

def set_powers(conf, power, laser_id):
    """
    Sets the power of a laser across all channels

    :param conf: A configuration object
    :param power: Power of the laser in [0, 100]
    :param laser_id: ID of the laser in Imspector (starting from 0)
    """
    for channel_id in range(microscope.get_num_channels(conf)):
        microscope.set_power(conf, power, laser_id, channel_id)

p_ex = partial(set_powers, laser_id=EXC_ID)
pdt = partial(microscope.set_dwelltime, channel_id=0)

# Channel 1 specific methods
ch1_signal_level = partial(microscope.set_rescue_signal_level, channel_id=0)

# Channel 2 specific methods
ch2_signal_level = partial(microscope.set_rescue_signal_level, channel_id=1)
ch2_strength = partial(microscope.set_rescue_strength, channel_id=1)
ch2_LTh_times = partial(microscope.set_LTh_times, channel_id=1)
ch2_LTh_thresholds = partial(microscope.set_LTh_thresholds, channel_id=1)
ch2_UTh_threshold = partial(microscope.set_UTh_threshold, channel_id=1)
ch2_p_sted = partial(microscope.set_power, laser_id=STED_ID, channel_id=1)
