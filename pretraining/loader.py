
import numpy
import h5py
import os
import torch

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def load_data(path, trial=0, scale=False, gridsearch=False, slc=slice(None, None)):
    """
    Loads the data from the given path

    :param path: A `str` of the path
    :param trial: An `int` of the number of path

    :returns : A `dict` of the X
               A `dict` of the y
    """
    if os.path.isfile(os.path.join(path, "optim.hdf5")):
        with h5py.File(os.path.join(path, "optim.hdf5"), "r") as file:
            if trial == "all":
                X, y = [], []
                for key in sorted(file["X"].keys(), key=lambda x : int(x)):
                    X.append(file["X"][key][slc])
                    y.append(file["y"][key][slc])
                X = numpy.concatenate(X, axis=0)
                y = numpy.concatenate(y, axis=0)
            else:
                X = file["X"][f"{trial}"][slc]
                y = file["y"][f"{trial}"][slc]
    else:
        X = numpy.loadtxt(os.path.join(path, f"X_{trial}.csv"), delimiter=",")
        y = numpy.loadtxt(os.path.join(path, f"y_{trial}.csv"), delimiter=",")
    if X.ndim == 1:
        X = X[:, numpy.newaxis]
    return {
        "X" : X, "y" : y
    }

def load_data_info(path):
    """
    Loads the data from the given path

    :param path: A `str` of the path
    :param trial: An `int` of the number of path

    :returns : A `dict` of the X
               A `dict` of the y
    """
    info = []
    if os.path.isfile(os.path.join(path, "optim.hdf5")):

        with h5py.File(os.path.join(path, "optim.hdf5"), "r") as file:
            for key in sorted(file["X"].keys(), key=lambda x : int(x)):
                info.append({
                    "trial" : key,
                    "shape" : file["X"][key].shape,
                    "path" : path
                })
    return info

def load_history(path, trial=0, length=1, scale=False, gridsearch=False):
    """
    Loads the data from the given path

    :param path: A `str` of the path
    :param trial: An `int` of the number of path

    :returns : A `dict` of the X
               A `dict` of the y
    """
    if os.path.isfile(os.path.join(path, "optim.hdf5")):
        with h5py.File(os.path.join(path, "optim.hdf5"), "r") as file:
            history = file["history"][f"{trial}"]
            X, y, ctx = [], [], []
            for i in range(length):
                if str(i) not in history:
                    break
                X.append(history[str(i)]["X"][()])
                y.append(history[str(i)]["y"][()])
                ctx.append(history[str(i)]["ctx"][()])
    X, y, ctx = numpy.array(X).squeeze(), numpy.array(y), numpy.array(ctx)
    return {
        "X" : X,
        "y" : y,
        "ctx" : ctx.squeeze()
    }

def indexitems(data, index):
    """
    Recursively iterates over a dict-like structure and indexes the arrays

    :param data: A `dict` of the data
    :param index: An `int` of the position at which to index the array

    :returns : A `dict` with indexed arrays
    """
    out = {}
    if isinstance(data, dict):
        for key, values in data.items():
            out[key] = indexitems(values, index)
    else:
        return data[index]
    return out

class ImageLoader(Dataset):
    def __init__(self, models, load_cache=False):
        """
        :param models: A `list` of model path
        """
        if isinstance(models, str):
            models = [models]

        self.models = models
        self.load_cache = load_cache

        self._data_info_cache = []
        self._data_cache = {}

        self._load_data_info()

    def _load_data_info(self):
        """
        Loads data information in cache
        """
        for model in tqdm(self.models, desc="Loading data info..."):
            info = load_data_info(model)
            self._data_info_cache.extend(info)

            if self.load_cache:
                for _info in tqdm(info, desc="Caching data...", leave=False):
                    cache_key = self._load_data(_info)

    def _load_data(self, info):
        """
        Loads data given some information. If required the data is loaded in cache

        :param info: A `dict` of the data to load

        :returns : A `str` of the cache key
        """
        data = load_data(info["path"], trial=info["trial"])
        history = load_history(info["path"], trial=info["trial"], length=info["shape"][0])

        cache_key = "-".join((info["path"], info["trial"]))
        if not self.load_cache:
            self._data_cache = {}
        self._data_cache[cache_key] = {
            "data" : data,
            "history" : history
        }
        return cache_key

    def get_data(self, index):
        """
        Gets the data at the given index

        :param index: An `int` of the position of the data

        :returns : A `dict` of the data
        """
        # Convert to index in cache
        cumsum = 0
        for info in self._data_info_cache:
            cumsum += info["shape"][0]
            if index < cumsum:
                dataset_index = index - (cumsum - info["shape"][0])
                break

        cache_key = "-".join((info["path"], info["trial"]))
        if cache_key not in self._data_cache:
            cache_key = self._load_data(info)

        return indexitems(self._data_cache[cache_key], dataset_index)

    def __getitem__(self, index):
        """
        Implements the `__getitem__` method of the `ImageLoader`

        :param index: A `int` of the index to retrieve from cache

        :returns : A `numpy.ndarray` of the input
                   A `numpy.ndarray` of the target
                   A `dict` of the history
        """
        data = self.get_data(index)
        X, y = data["data"]["X"], data["data"]["y"]
        history = data["history"]

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        history = {
            key : torch.tensor(value, dtype=torch.float32) for key, value in history.items()
        }

        return X, y, history

    def __len__(self):
        """
        Implements the `__len__` method of the `ImageLoader`

        :returns : An `int` of the number of possible options
        """
        return sum([
            info["shape"][0] for info in self._data_info_cache
        ])

class SubsetDataset:
    def __init__(self, dataset, indices):

        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):

        index = self.indices[index]
        return self.dataset[index]

    def __len__(self):
        return len(self.indices)

def get_loader(models, split_train_valid=1., **kwargs):
    """
    Creates a `DataLoader` instance

    :param models: A `list` of model path
    """
    dataset = ImageLoader(models, load_cache=True)
    if split_train_valid < 1.:
        indices = numpy.random.permutation(numpy.arange(len(dataset)))
        length = int(split_train_valid * len(indices))
        train_indices, valid_indices = indices[:length], indices[length:]
        train_loader = DataLoader(
            SubsetDataset(dataset, train_indices), **kwargs.get("dataloader_opts", {"batch_size": 16, "shuffle": True})
        )
        valid_loader = DataLoader(
            SubsetDataset(dataset, valid_indices), **kwargs.get("dataloader_opts", {"batch_size": 16, "shuffle": True})
        )
        return train_loader, valid_loader
    return DataLoader(
        dataset, **kwargs.get("dataloader_opts", {"batch_size": 16, "shuffle": True})
    )

if __name__ == "__main__":

    import yaml

    model_name = [
        "../data/contextual-image/20221028-163625_5a29e1c1_DyMIN_prefnet_ContextualImageLinTSDiag",
        # "../data/contextual-image/20221028-163625_5a29e1c1_DyMIN_prefnet_ContextualImageLinTSDiag",
    ]

    # config = yaml.load(open(os.path.join(model_name, "config.yml"), "r"), Loader=yaml.Loader)
    # X, y = load_data(model_name, trial=0)
    # history = load_history(config, model_name, trial=0)

    loader = ImageLoader(model_name)
    loader[0]
