from MoASubmitter import *

import numpy as np


def _save_cache(X, Y, pathx, pathy):
    assert np.shape(X)[0] == np.shape(Y)[0]

    print("[*] Saving data...")
    np.save(pathx, X, allow_pickle=False)
    np.save(pathy, Y, allow_pickle=False)

    print("[*] Self-checking..")
    X_saved = np.load(f'{pathx}.npy', allow_pickle=False)
    Y_saved = np.load(f'{pathy}.npy', allow_pickle=False)

    assert (X_saved == X).all()
    assert (Y_saved == Y).all()

    print("[+] Cache and self-check completed successfully")


def build_cache():
    print("[*] Extracting data...")
    X, Y = extract_xy('./data/train_features.csv', './data/train_targets_scored.csv')
    _save_cache(X, Y, './data/cache/X', './data/cache/Y')


def build_nfkbi_cache():
    print("[*] Extracting data...")
    X, Y = extract_xy_nfkbi('./data/train_features.csv', './data/train_targets_scored.csv')
    _save_cache(X, Y, './data/cache/X_nfkbi', './data/cache/Y_nfkbi')


def build_noscore_cache():
    print("[*] Extracting data...")
    X, Y = extract_xy('./data/train_features.csv', './data/train_targets_nonscored.csv')
    _save_cache(X, Y, './data/cache/X', './data/cache/Y_nonscored')


def load_cache(x_cache_path, y_cache_path):
    X = np.load(x_cache_path, allow_pickle=False)
    Y = np.load(y_cache_path, allow_pickle=False)

    return X, Y


if __name__ == '__main__':
    #build_nfkbi_cache()
    build_noscore_cache()