"""
Functions necessary on the kaggle side

"""

import os

import pandas as pd
import numpy as np

from tensorflow import keras

if 'USER' in os.environ and os.environ['USER'] == 'isears':
    DATA_ROOT = './data'
    MODEL_ROOT = './models'
else:
    DATA_ROOT = '/kaggle/input/lish-moa'
    MODEL_ROOT = '/kaggle/input/moa-prediction-models'


def numpy_to_submission(arr, ids, columns):
    assert np.shape(arr)[1] == (len(columns) - 1)
    total_rows = len(ids)
    cache = list()

    for row_idx, id in enumerate(ids):
        curr_cache = dict()
        curr_cache['sig_id'] = id

        for column_idx, cname in enumerate(columns[1:]):  # Skip sig_id
            curr_cache[cname] = arr[row_idx][column_idx]

        cache.append(curr_cache)

    out_df = pd.DataFrame.from_dict(cache)
    out_df.to_csv('./submission.csv', index=False)

def xrow_to_numpy(row):
    # two columns need to be converted to floats: cp_type and cp_dose
    clean_row = row.copy(deep=True)

    if row['cp_type'] == 'trt_cp':
        clean_row['cp_type'] = 1.
    elif row['cp_type'] == 'ctl_vehicle':
        clean_row['cp_type'] = 0.
    else:
        raise ValueError(f'Unrecognized cp_type: {row["cp_type"]}')

    if row['cp_dose'] == 'D1':
        clean_row['cp_dose'] = 0.
    elif row['cp_dose'] == 'D2':
        clean_row['cp_dose'] = 1.
    else:
        raise ValueError(f'Unrecognized cp_dose: {row["cp_dose"]}')

    clean_row = clean_row.drop('sig_id')

    return clean_row.to_numpy(dtype=float)


def yrow_to_numpy(row):
    clean_row = row.copy(deep=True)
    clean_row = clean_row.drop('sig_id')

    return clean_row.to_numpy(dtype=float)


def extract_xy(x_path, y_path):
    df_x = pd.read_csv(x_path)
    df_y = pd.read_csv(y_path)

    x_count = len(df_x.index)
    y_count = len(df_y.index)

    assert x_count == y_count

    feature_count = len(df_x.columns) - 1  # sig_id not included as feature
    label_count = len(df_y.columns) - 1  # sig_id not included as label

    X = np.zeros((x_count, feature_count))
    Y = np.zeros((y_count, label_count))

    idx = 0
    for sig_id in df_x.sig_id:
        row_x = df_x[df_x.sig_id == sig_id]
        row_y = df_y[df_y.sig_id == sig_id]
        assert len(row_x.index) == 1  # sig_id should be unique
        assert len(row_y.index) == 1
        row_x = row_x.iloc[0]
        row_y = row_y.iloc[0]

        X[idx] = xrow_to_numpy(row_x)
        Y[idx] = yrow_to_numpy(row_y)
        idx += 1

    return X, Y


def extract_xy_nfkbi(x_path, y_path):
    df_x = pd.read_csv(x_path)
    df_y = pd.read_csv(y_path)

    x_count = len(df_x.index)
    y_count = len(df_y.index)

    assert x_count == y_count
    feature_count = len(df_x.columns) - 1  # sig_id not included as feature

    X = np.zeros((x_count, feature_count))
    Y = np.zeros((y_count, 1)) # Only getting nfkb_inhibitor

    idx = 0
    for sig_id in df_x.sig_id:
        row_x = df_x[df_x.sig_id == sig_id]
        row_y = df_y[df_y.sig_id == sig_id]
        assert len(row_x.index) == 1  # sig_id should be unique
        assert len(row_y.index) == 1
        row_x = row_x.iloc[0]
        row_y = row_y.iloc[0]

        X[idx] = xrow_to_numpy(row_x)
        Y[idx] = row_y['nfkb_inhibitor']
        idx += 1

    return X, Y


def extract_x(x_path):
    df_x = pd.read_csv(x_path)
    x_count = len(df_x.index)

    feature_count = len(df_x.columns) - 1  # sig_id not included as feature

    X = np.zeros((x_count, feature_count))

    idx = 0
    for sig_id in df_x.sig_id:
        row_x = df_x[df_x.sig_id == sig_id]
        assert len(row_x.index) == 1  # sig_id should be unique
        row_x = row_x.iloc[0]

        X[idx] = xrow_to_numpy(row_x)
        idx += 1

    return X, df_x.sig_id


if __name__ == '__main__':
    print("[*] Loading test data...")
    X, sig_ids = extract_x(f'{DATA_ROOT}/test_features.csv')
    y_columns = pd.read_csv(f'{DATA_ROOT}/sample_submission.csv').columns
    print(f'\t{np.shape(X)[0]} training examples with {np.shape(X)[1]} features')

    print("[*] Loading model...")
    model = keras.models.load_model(f'{MODEL_ROOT}/model.h5')
    print("[+] Model loaded")

    print("[*] Making predictions...")
    y_hat = model.predict(X)
    print("[+] Prediction matrix generated:")
    print(y_hat)

    print("[*] Saving predictions...")
    numpy_to_submission(y_hat, sig_ids, y_columns)
    print("[+] Predictions saved")