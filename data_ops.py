''' Data preparation operations '''

import pandas as pd
import os
import zlib
import zipfile

from sklearn.model_selection import train_test_split

from my_feature_engine import *


def read_compress(path_to_data, path_to_archive = 'compressed_data.zip'):
    """ Read data into memory; compress and delete raw file."""

    df = pd.read_csv(path_to_data)

    with zipfile.ZipFile(path_to_archive, 'w') as zf:
        zf.write(path_to_data, arcname = 'password_dataset.csv', compress_type = zipfile.ZIP_DEFLATED)
        zf.close()

    os.remove(path_to_data)

    return df, path_to_archive


def decompress(path_to_archive, path_to_data):
    """ Decompress archived dataset. """

    with zipfile.ZipFile(path_to_archive, 'r') as zf:
        try:
            zf.read(path_to_archive)
        except KeyError:
            print(f'File not found! Ensure archive at {path_to_archive} is available.')
    return


def generate_features(df: pd.DataFrame, p1: str) -> pd.DataFrame:
    """ Generate features for password(s). """

    assert type(df) in [pd.Series, pd.DataFrame], 'df must be pandas.DataFrame or pandas.Series object.'
    
    data = pd.DataFrame()
    df = df.dropna()

    if type(df) == pd.DataFrame:
        cols = df.columns

        if 'strength' in cols:
            target = df['strength']

        df = df.loc[:, 'password']
    else:
        cols = None

    data['num_upper'] = df.apply(count_characters, convert_dtype=False, **{'alpha': True, 'upper': True})
    data['num_lower'] = df.apply(count_characters, convert_dtype=False, **{'alpha': True, 'upper': False})
    data['num_numbers'] = df.apply(count_characters, convert_dtype=False, **{'alpha': False, 'upper': True})

    data['password_length'] = df.apply(lambda x: len(x))
    data['consecutive_num_pairs'] = df.apply(get_consecutive_numbers)
    data['consec_kboard_chars'] = df.apply(match_re, convert_dtype=True,
                                           **dict(zip(('p1', 'p2'), (generate_patterns(p1)))))

    data['num_puncts'] = df.apply(count_punct)
    data['num_consec_punct_pairs'] = df.apply(get_consecutive_punct)
    data['num_consec_space_pairs'] = df.apply(get_consecutive_whitespaces)

    data['num_whitespaces'] = df.apply(count_whitespaces)

    return data if cols is None else (data, target)


def split_data(X, y, split_size=0.2):
    """ Split dataset for model evaluation and testing. """

    X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=split_size, stratify=y)

    return X_1, X_2, y_1, y_2
