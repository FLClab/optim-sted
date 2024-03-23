# Contextual-Bandit optimization for STED microscopy

This folder contains the code to optimize the performance of stimulated emission depletion (STED) microscopy using a contextual-bandit algorithm in a simulated environment.

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

**Base configuration MODEL**

```yaml
regressor_args:
  default:
    share_ctx: True # If `True`, the context encoder is shared between the objectives
    teacher_opts: # Experimental; Creates a student-teacher configuration for training
      use: False # If `True`, the a teacher model is used
      alpha: 1.0e-3 # The rate of update of the teacher for the exponential moving average
    every-step-decision: False # If `True`, the decision is made at every step; Can be used in cases of multuple acquisitions per step
```

### Experiment configuration

The experiment configuration file is a YAML file that contains the configuration for the optimization process. The experiment configuration file is passed as an argument to the `run.py` command line interface. The experiment configuration file is a YAML file that contains the configuration for the optimization process. The experiment configuration file is passed as an argument to the `run.py` command line interface. The experiment configuration file should inherit the base configuration file.