
import os
import numpy
import yaml
import microscope
import user
import utils

with open(os.path.join("configs", "dymin-configs.yml"), "r") as file:
    config = yaml.load(file)

for channel in config["ExpControl"]["measurement"]["channels"]:
    print(channel["rescue"].keys())
print()
