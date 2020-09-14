"""
Functions necessary on the kaggle side

"""


import pandas as pd

def extract_features(path):
    df = pd.read_csv(path)

    X = df
    return X


if __name__ == '__main__':
    quit()