
from functools import partial

from abberior import microscope

EXC_ID = 5
STED_ID = 6

if isinstance(EXC_ID, (list, tuple)):
    ch1_p_ex = partial(microscope.set_power, laser_id=EXC_ID[0], channel_id=0)
    ch1_p_sted = partial(microscope.set_power, laser_id=STED_ID, channel_id=0)
    ch1_linestep = partial(microscope.set_linestep, step_id=0)

    ch2_p_ex = partial(microscope.set_power, laser_id=EXC_ID[1], channel_id=1)
    ch2_p_sted = partial(microscope.set_power, laser_id=STED_ID, channel_id=1)
    ch2_linestep = partial(microscope.set_linestep, step_id=1)

else:
    p_ex = partial(microscope.set_power, laser_id=EXC_ID, channel_id=0)
    pdt = partial(microscope.set_dwelltime, channel_id=0)
    p_sted = partial(microscope.set_power, laser_id=STED_ID, channel_id=0)
    linestep = partial(microscope.set_linestep, step_id=0)
    spectral_min = partial(microscope.set_spectral_min, channel_id=0)
    spectral_max = partial(microscope.set_spectral_max, channel_id=0)
