from MoASubmitter import *

import numpy as np


def build_cache():
    print("[*] Extracting data...")
    X, Y = extract_xy('./data/train_features.csv', './data/train_targets_scored.csv')
    assert np.shape(X)[0] == np.shape(Y)[0]

    print("[*] Saving data...")
    np.save('./data/cache/X', X, allow_pickle=False)
    np.save('./data/cache/Y', Y, allow_pickle=False)

    print("[*] Self-checking..")
    X_saved = np.load('./data/cache/X.npy', allow_pickle=False)
    Y_saved = np.load('./data/cache/Y.npy', allow_pickle=False)

    assert (X_saved == X).all()
    assert (Y_saved == Y).all()

    print("[+] Cache and self-check completed successfully")


def load_cache(x_cache_path, y_cache_path):
    X = np.load(x_cache_path, allow_pickle=False)
    Y = np.load(y_cache_path, allow_pickle=False)

    return X, Y


if __name__ == '__main__':
    build_cache()