import pandas as pd
from pathlib import Path

def load_data(data_dir = "data/raw/synthetic_coffee_health_10000.csv"):
#
#   Loads csv data
#
#   args :
#       data_dir : path to the data directory
# 
#   returns :
#       pandas dataFrame
#

    path = Path(data_dir)
    data = pd.read_csv(path)

    return data


def main():
    #
    #
    #
    print('use load_data() function to load the data ' \
    '\nsynthetic_coffee_health_10000.csv will be automatically loaded if no path is specified')



