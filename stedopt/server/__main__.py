
import argparse
from .application import create_app

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", required=True, type=str, help="Log directory of the server")
args = parser.parse_args()

app = create_app(args.logdir)
app.run_server(debug=True)
