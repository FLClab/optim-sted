# stedopt-server

Provides a server for the ``optim-sted`` package. This server is used to display the results of the optimization process in a web interface.

*This code is still experimental.*

## Installation

After cloning the repository, install the package in editable mode with the following command:

```bash
pip install -e "./optim-sted[server]"
```

## Usage

To start the server, run the following command:

```bash
python -m stedopt.server --logdir <path-to-logdir>
```