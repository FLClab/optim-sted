
import numpy
import os, glob
import pickle
import h5py

PATH = "../../data"

if __name__ == "__main__":

    # # Resolution, Bleach, Squirrel
    # directories = [
    #     "20210618-145645_optim_1params_3objectives_degree4",
    #     "20210619-071126_optim_1params_3objectives_degree4",
    #     "20210621-095150_optim_2params_3objectives_degree4",
    #     "20210622-075840_optim_3params_3objectives_degree4"
    # ]
    #
    # X, y = [], []
    # for directory in directories:
    #     filenames = glob.glob(os.path.join(PATH, directory, "selected_indexes", "**/*.npz"), recursive=True)
    #     for filename in filenames:
    #
    #         data = numpy.load(filename)
    #         points_arr2d, y_samples, index = data["points_arr2d"], data["y_samples"], data["index"]
    #
    #         pareto = filename.replace("selected_indexes", "pareto_indexes")
    #         pareto = pareto.replace(".npz", ".csv")
    #         pareto_indexes = numpy.loadtxt(pareto, delimiter=",").astype(int)
    #         if pareto_indexes.ndim < 1:
    #             pareto_indexes = pareto_indexes[numpy.newaxis]
    #
    #         # Fix for squared error
    #         # y_samples[2] = y_samples[2] ** 2
    #
    #         X.append(y_samples.squeeze(axis=-1).transpose())
    #         y.append(index.item())
    #         # X.append(points_arr2d)
    #         # y.append(pareto_indexes[index])
    #
    # pickle.dump((X, y), open(os.path.join(PATH, "3objs-ResolutionBleachSquirrel-pref.pkl"), "wb"))
    #
    # X, y = pickle.load(open(os.path.join(PATH, "3objs-ResolutionBleachSquirrel-pref.pkl"), "rb"))
    # print(len(X), len(y))

    # # Resolution, Squirrel, Bleach
    # directories = [
    #     "20220629-090447_DyMIN_optim_LinTSDiag", # DyMIN, 3 parameters
    #     "20220630-080827_DyMIN_optim_sklearn_GP", # DyMIN, 3 parameters
    #     "20220629-091416_DyMIN_optim_LinTSDiag", # DyMIN, 7 parameters
    #     "20220630-080900_RESCue_optim_LinTSDiag", # RESCue, 6 parameters
    #     "20221208-user-click" # User clicks
    # ]
    #
    # X, y = [], []
    # for directory in directories:
    #     with h5py.File(os.path.join(PATH, directory, "optim.hdf5"), "r") as file:
    #         y_samples, selected_indexes = file["y_samples"], file["selected_indexes"]
    #         for repetition in selected_indexes.keys():
    #             for iteration in selected_indexes[repetition].keys():
    #                 samples = y_samples[repetition][iteration][()]
    #                 selected = selected_indexes[repetition][iteration][1] # Keeps the real selected item
    #
    #                 X.append(samples.squeeze(axis=-1).transpose())
    #                 y.append(selected)
    #
    # pickle.dump((X, y), open(os.path.join(PATH, "20221208-3objs-ResolutionSquirrelBleach-pref.pkl"), "wb"))
    #
    # X, y = pickle.load(open(os.path.join(PATH, "20221208-3objs-ResolutionSquirrelBleach-pref.pkl"), "rb"))
    # print(len(X), len(y))

    # Resolution, Bleach, SNR
    directories = [
        "20230207-110427_f9c9e1af_STED_optim_sklearn_GP",
        "20230208-063939_48e501c3_STED_optim_sklearn_GP",
        "20230208-081224_2f9ff214_STED_optim_sklearn_GP",
        "20230713-100507_3147a732_STED_optim_LinTSDiag",
        "20230713-134549_b2a6e9e6_STED_optim_LinTSDiag",
        "20230713-145403_5d9e1aeb_STED_optim_LinTSDiag",
        "20230714-065742_e253b501_STED_prefnet_LinTSDiag",
        "20230714-073717_bb985ff8_STED_prefnet_LinTSDiag",
        "20230714-075932_5c63f675_STED_prefnet_LinTSDiag",
        "20230714-083330_86fd373f_STED_prefnet_LinTSDiag",
        "20230714-090000_54c1c014_STED_prefnet_LinTSDiag",
        "20230714-093332_74e9eebb_STED_prefnet_LinTSDiag",
        "20230714-100147_b7259501_STED_prefnet_LinTSDiag",
        "20230714-140501_9b5f7875_STED_prefnet_sklearn_GP",
        "20230714-143224_32c74b0a_STED_prefnet_sklearn_GP",
    ]

    X, y = [], []
    for directory in directories:
        with h5py.File(os.path.join(PATH, directory, "optim.hdf5"), "r") as file:
            y_samples, selected_indexes = file["y_samples"], file["selected_indexes"]
            for repetition in selected_indexes.keys():
                for iteration in selected_indexes[repetition].keys():
                    samples = y_samples[repetition][iteration][()]
                    selected = selected_indexes[repetition][iteration][1] # Keeps the real selected item

                    X.append(samples.squeeze(axis=-1).transpose())
                    y.append(selected)

    pickle.dump((X, y), open(os.path.join(PATH, "20230714-3objs-ResolutionBleachSNR-pref.pkl"), "wb"))

    X, y = pickle.load(open(os.path.join(PATH, "20230714-3objs-ResolutionBleachSNR-pref.pkl"), "rb"))
    print(len(X), len(y))

    # This assumes that you have clone the STEDPreferenceSNN repository
    # python train.py --margin 0.1 --batch-size 128 --nb-epochs 100 --random-state 42 --cuda /home-local/optim-sted/3objs-ResolutionBleachSNR-pref.pkl /home-local/optim-sted/prefnet
