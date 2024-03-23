# Bandit optimization for STED microscopy

This folder contains the code to optimize the performance of stimulated emission depletion (STED) microscopy using a bandit algorithm in a simulated environment.

## Configuration

The configuration file is a YAML file that contains the configuration for the optimization process. The configuration file is passed as an argument to the `run.py` command line interface.

### Base configuration

We provide a base configuration file that can be used as a template for the optimization process. The base configuration file is a YAML file that contains the configuration for the optimization process. The base configuration file is passed as an argument to the `run.py` command line interface.

**Base configuration**
```yaml
datamap_opts: # This is the configuration for the datamap
  mode: generate # The mode can be either 'generate' or 'real' or 'real-complete'
  shape: # The shape of the datamap
    - 96
    - 96
  sources: # If `generate`; sets the number of sources
    - 100
    - 250
  molecules: 40 # Multiplication (mean) factor for the number of molecules
  molecules_scale: 0.1 # Standard deviation for the number of molecules (mean +/- scale)
  shape_sources: # If `generate`; sets the shape of the sources
    - 2
    - 2
  random_state: 42 # Random state for reproducibility
  sequence_per_trial: false # If `true`, each trial will have its own sequence of datamaps
  path: # If `real` or `real-complete`, the path to the datamaps
    - ../data/datamap/PSD95-Bassoon
articulation_opts: # This is the configuration for the articulation used with `--prefart optim`
  borders: # The axis of the point cloud, one per objectives
    mins: # The minimum values for the axes
      - 40.0
      - 0.0
      - 0.0
    maxs: # The maximum values for the axes
      - 180.0
      - 1.0
      - 1.0
  cmap: rainbow
default_values_dict: # The default values for the acquisition parameters
  num_acquisition: 1 # The number of acquisitions on the same datamap
  p_ex: 2.0e-06 # The default excitation power
  p_sted: 150.0e-3 # The default STED power
  pdt: 10.0e-6 # The default pixel dwell time
  pixelsize: 2.0e-08 # The default pixel size
knowledge_opts: # The configuration for the prior domain knowledge
  use: true # If `true`, the prior knowledge is used
  update_posterior: true # If `true`, the posterior is updated during each step of domain knowledge
  num_random_samples: 3 # The number of random samples to be acquired using domain knowledge
  mode: expert # The mode of domain knowledge; can be `expert` or `pareto` or `random`
  expert_knowledge: "DyMIN" # If `expert`, the filename of the domain knowledge
  pareto_samples: # If `pareto`, the filename of the Pareto front
    - ../data/exhaustive/20220906-161541_DyMIN_NSGA_None
routine_opts: # The configuration for the routine and the fluorophores
  use: false # If `true`, the routine bandit configuration is used
  routine_file: "complete-routines.json" # The filename of the fluorophores parameters
  mode: select # The mode of the routine; can be `select` or `random` or `default`
  routine_id: 2 # If `select`, the ID of the routine as an index value
  models: null # If `use=true`, the models to be used
hide_acquisition: true # If `true`, the selected imaging parameters and imaging objectives are hidden
rescale_opts: # The configuration for the rescaling of the objectives
  use: false # If `true`, the objectives are rescaled
  mins: # The minimum values for the rescaling of each objectives
    - 40.0
    - 0.0
    - 0.0
  maxs: # The maximum values for the rescaling of each objectives
    - 180.0
    - 1.0
    - 1.0
model_config: # The configuration for the PrefNet model
  nb_obj: 3 # The number of objectives
  middle_size: 10 # The size of the middle layer of the PrefNet model
  path: ../data/prefnet/2022-12-08-13-37-53/weights.t7 # The path to the weights of the PrefNet model
  config_path: ../data/prefnet/2022-12-08-13-37-53/config.json # The path to the configuration of the PrefNet model
n_divs_default: 10 # The default number of divisions for the `grid` mode
nbre_trials: 50 # The number of trials for the optimization; a trial corresponds to one imaging session
multiprocess_opts: # The configuration for the multiprocessing
  num_processes: 0 # The number of processes to be used; A process per trial is used if `>0`
  verbose: True # If `True`, the multiprocessing is verbose
obj_names: # The names of the objectives; Corresponds to keys in `defaults.obj_dict`
- Resolution
- Squirrel
- Bleach
optim_length: 200 # The number of iterations for the optimization
params_conf: # The configuration for the acquisition parameters of the confocal images
  p_ex: 1.0e-06 # The default excitation power
  p_sted: 0.0 # The default STED power
  pdt: 10.0e-6 # The default pixel dwell time
pareto_opts: # The configuration for the Pareto 
  mode: nsga # The mode of the Pareto front; can be `nsga` or `default` or `grid` or `uniform`
random_state: null # Random state for reproducibility
save_folder: ../data # The folder where the results are saved
with_time: true # If `true`, the time is used during the optimization; We always want to minimize the time
```

**Base configuration MICROSCOPE**

```yaml
microscope: STED # The name of the microscope
obj_names: # The names of the objectives; In STED, the objective Squirrel is replaced by SNR
- Resolution
- Bleach
- SNR
model_config: # In STED, the PrefNet model is differen
  path: ../data/prefnet/2023-07-14-14-23-36/weights.t7
  config_path: ../data/prefnet/2023-07-14-14-23-36/config.json
```

**Base configuration MODEL**

```yaml
_BASE_: BASE_config.yml # The name of the base configuration file
pareto_opts:
  mode: nsga # `nsga` is used with LinTSDiag or NeuralTS; `default` is used with gp (Kernel-TS)
regressor_args: # The configuration for the regressor
  Bleach: # Overwrites the default values for the Bleach objective; See `banditopt.models` for parameters
    _lambda: 0.1
    nu: 0.01
  Resolution:
    _lambda: 0.1
    nu: 0.01
    min_features:
      - 40.
    max_features:
      - 250.
  SNR:
    _lambda: 0.1
    nu: 0.01
  Squirrel:
    _lambda: 0.1
    nu: 0.01
  FFTMetric:
    _lambda: 0.1
    nu: 0.1    
  default: # Default values for the regressor
    n_features: 3 # The number of features; Corresponds to the number of objectives
    n_hidden_dim: 32 # The number of hidden dimensions in the neural network
    _lambda: 0.0001 # The regularization parameter of LinTSDiag and NeuralTS
    nu: 0.01 # The exploration parameter of LinTSDiag and NeuralTS
    learning_rate: 1.0e-3 # The learning rate of the neural network
    min_features: # The minimum values for the features; Used to rescale the objectives
      - 0.
    max_features: # The maximum values for the features; Used to rescale the objectives
      - 1.0
    update_exploration: false # If `true`, the exploration parameter is updated using 1/sqrt(t)
    pretrained_opts: # The configuration for the pretrained model
        udpate: True # If `True`, the pretrained model is updated
        path: "../data/pretrained/20221128-112415_2b6deef8_DyMIN_None_LinTSDiag" # The path to the pretrained model
        load_all: False # If `True`, all the weights are loaded
        trial: 0 # The trial of the pretrained model to choose
        use: False # If `True`, the pretrained model is used
regressor_name: LinTSDiag # The name of the regressor; Corresponds to `defaults.regressor_dict`
```

### Experiment configuration

The experiment configuration file is a YAML file that contains the configuration for the optimization process. The experiment configuration file is passed as an argument to the `run.py` command line interface. The experiment configuration file is a YAML file that contains the configuration for the optimization process. The experiment configuration file is passed as an argument to the `run.py` command line interface. The experiment configuration file should inherit the base configuration file.

**Experiment configuration**
```yaml
_BASE_: # A `list` of base configuration files
- BASE_config_sted.yml
- BASE_config_LinTSDiag.yml
param_names: # The names of the acquisition parameters
- p_sted
- p_ex
- pdt
random_state: 42 # Random state for reproducibility
x_maxs: # The maximum values for the acquisition parameters; Same oreder as `param_names`
- 300.0e-3
- 2.0e-6
- 30.0e-6
x_mins: # The minimum values for the acquisition parameters; Same oreder as `param_names`
- 0.
- 0.
- 0.
regressor_args: # Depending on the experiment, the parameters can be different (e.g. different number of objectives)
  default:
    n_features: 3

```