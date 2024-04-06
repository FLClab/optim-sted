# Installation

This is the section to follow to install on a normal machine.

Create python environment
```bash
conda create --name optim-sted python=3.10
conda activate optim-sted
```

Clone necessary repositories
```bash
git clone https://github.com/FLClab/optim-sted.git
```

Install a local version of the package
```bash
pip install ./optim-sted
```

## Abberior Microscope

Install ``specpy`` from a provided wheel with the Imspector software
```bash
pip install <PATH>/specpy-1.2.3-cp39-cp39-win_amd64.whl
```

Run code example
```bash
cd optim-sted/abberior
python run.py --config ./configs/config_abberior-dymin_7params_random.yml --dry-run
```

## Slurm Installation

Create python environment
```
module load python/3.10
module load scipy-stack

virtualenv --no-download ~/venvs/optim-sted
source ~/venvs/optim-sted/bin/activate
pip install --no-index --upgrade pip
```

Clone necessary repositories
```
git clone https://github.com/FLClab/optim-sted.git
```

Install dependencies
```
pip install -e optim-sted --no-index
```
