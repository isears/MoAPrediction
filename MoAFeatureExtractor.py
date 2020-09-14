import pandas as pd
import numpy as np


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
    # X, Y = extract_xy(
    #     './data/train_features.csv',
    #     './data/train_targets_scored.csv'
    # )
    #
    # print(np.shape(X))
    # print(X)
    # print(np.shape(Y))
    # print(Y)

    X, sig_ids = extract_x('./data/test_features.csv')
    print(X)
    print(sig_ids)
