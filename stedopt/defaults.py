
from banditopt import algorithms, objectives

P_STED = 150.0e-3
P_EX = 2.0e-6
PDT = 10.0e-6

# LASER_EX = {"lambda_" : 488e-9}
# LASER_STED = {"lambda_" : 575e-9, "zero_residual" : 0.01}
LASER_EX = {"lambda_" : 635e-9}
LASER_STED = {"lambda_" : 750e-9, "zero_residual" : 0.005, "anti_stoke": False}
DETECTOR = {"noise" : True, "background" : 0.5 / PDT}
OBJECTIVE = {
    "transmission" : {488: 0.84, 535: 0.85, 550: 0.86, 585: 0.85, 575: 0.85, 635: 0.84, 690: 0.82, 750: 0.77, 775: 0.75}
}

FLUO = { # ATTO647N
    "lambda_": 6.9e-7,
    "qy": 0.65,
    "sigma_abs": {
        635: 2.14e-20,
        750: 3.5e-25
    },
    "sigma_ste": {
        750: 3.0e-22
    },
    "tau": 3.5e-9,
    "tau_vib": 1e-12,
    "tau_tri": 0.0000012,
    "k0": 0,
    "k1": 1.3e-15,
    "b": 1.6,
    "triplet_dynamics_frac": 0
}

action_spaces = {
    "p_sted" : {"low" : 0., "high" : 350.0e-3},
    "p_ex" : {"low" : 0., "high" : 10.0e-6},
    "pdt" : {"low" : 0.0e-6, "high" : 60.0e-6},
}

# Define the objectives and regressors here
obj_dict = {
    "SNR" : objectives.Signal_Ratio(75),
    "Bleach" : objectives.Bleach(),
    "Resolution" : objectives.Resolution(pixelsize=20e-9, res_cap=225),
    # "Resolution" : objectives.FWHMResolution(pixelsize=20e-9),
    "Squirrel" : objectives.Squirrel(normalize=True, use_foreground=True),
    "FFTMetric" : objectives.FFTMetric(),
    "Crosstalk" : objectives.Crosstalk()
}
regressors_dict = {
    "sklearn_BayesRidge" : algorithms.sklearn_BayesRidge,
    "sklearn_GP" : algorithms.sklearn_GP,
    "LinTSDiag" : algorithms.LinearBanditDiag,
    "NeuralTS" : algorithms.NeuralTS,
    "LSTMLinTSDiag" : algorithms.LSTMLinearBanditDiag,
    "ContextualLinTSDiag" : algorithms.ContextualLinearBanditDiag,
    "ContextualNeuralTS" : algorithms.ContextualNeuralTS,
    "ContextualImageLinTSDiag" : algorithms.ContextualImageLinearBanditDiag,
    "ContextualImageNeuralTS" : algorithms.ContextualImageNeuralTS,
}
