import copy
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import scipy as sc
import json
import yaml 

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

print(config['simulation'])
aux = config['simulation']

for chave, valor in aux.items():
    if isinstance(valor, float):
        aux[chave] = valor * 2

if os.path.exists('configDef.yml'):
        os.remove('configDef.yml')

with open('configDef.yml', 'x') as f:
    config['simulation'] = aux
    yaml.dump(config, f, default_flow_style=True)
