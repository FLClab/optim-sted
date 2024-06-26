
from setuptools import find_packages
from distutils.core import setup

setup(
    name="optim-sted",
    version="0.1",
    author="Anthony Bilodeau",
    install_requires=[
        "bandit-optimization",
        "pysted",
        "abberior-sted",
        "h5py",
        "pyyaml",
        "metrics @ git+https://github.com/FLClab/metrics.git",
        "numpy",
        "matplotlib",
        "torch",
        "torchvision",
        "seaborn",
        "umap-learn",
    ],
    extras_require={
    "server": ["dash", "plotly", "dash_bootstrap_components", "dash_renderjson", "dash_mantine_components"],
    },
    packages=find_packages(include=["stedopt", "stedopt.*", "stedopt.server"]),
    include_package_data=True,
)
