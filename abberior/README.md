# Contextual-Bandit optimization for STED microscopy

This folder contains the code to optimize the performance of stimulated emission depletion (STED) microscopy using a bandit or contextual-bandit algorithm on a real STED microscope.

## Configuration

The configuration file is a YAML file that contains the configuration for the optimization process. The configuration file is passed as an argument to the `run.py` command line interface. We herein present only the configuration that are different from the bandit optimization. Please refer to the [bandit optimization](../bandit/README.md) for the common configuration.

### Base configuration

We provide a base configuration file that can be used as a template for the optimization process. The base configuration file is a YAML file that contains the configuration for the optimization process. The base configuration file is passed as an argument to the `run.py` command line interface.

**Base configuration**

```yaml
ctx_opts: # The configuration for the context
  use_ctx: true # If `true`, the context is used
  ctx_x_maxs: # The maximum values for the context; Used to rescale the context
  - 200.
  ctx_x_mins: # The minimum values for the context; Used to rescale the context
  - 0.
  mode: quantile # The mode of the context; Can be `mean` or `quantile` or `image`
```

**Base configuration MICROSCOPE**

```yaml
microscope_conf: # The configuration for the microscope
    mode: AbberiorSTED # The mode of the microscope; See `stedopt.tools.MicroscopeConfigurator` for implementation details
    acquisition-mode: "normal" # The acquisition mode of the microscope; See `stedopt.tools.MicroscopeConfigurator` for implementation details
```

### Experiment configuration

The experiment configuration file is a YAML file that contains the configuration for the optimization process. The experiment configuration file is passed as an argument to the `run.py` command line interface. The experiment configuration file is a YAML file that contains the configuration for the optimization process. The experiment configuration file is passed as an argument to the `run.py` command line interface. The experiment configuration file should inherit the base configuration file.

## Microscope defaults

The `MicroscopeConfigurator` requires to set the default values of each microscope modalities. This is handled by modifying the values contained in `default/Abberior<MODALITY>.py`. This file defines the name of the parameters than can be optimized and how they are modified on the real microscope using `functools.partial`.

Other modalities or functions can be implemented if required.

## Launch optimization

There are multiple files that can be used to launch an experiment. The reason for these different files are simply to handle different specific experimental conditions. 

### `run.py`

This is the main file that should be used for most acquisition to optimize the imaging parameters of different modalities (STED, DyMIN, RESCue). 

### `run-multicolor.py`

This file handles a multicolor optimization. Currently, the imaging objectives from each color are averaged together to facilitate the selection from the imaging objective space by the microscopist. 

### `run3D.py`

This file handles the optimization of 3D acquisitions. Currently, the imaging objectives are calculated only on the central frame of the acquired stack.

### `run-manual.py`

This file does not use the bandit optimization framework to optimize the microscopy task. All parameters are selected by the microscopist. 
