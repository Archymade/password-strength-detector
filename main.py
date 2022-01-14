""" Train model from command line. """

import argparse
import os
import time
import pickle
import random

from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from data_ops import read_compress
from data_ops import generate_features
from data_ops import split_data

from metrics import get_precision_score, get_accuracy_score
from metrics import get_recall_score, get_f1_score

from model_utils import import_model, train_model
from viz_utils import visualize_confusion_matrix


def configure_args():
    """ Configure cli arguments. """

    args = argparse.ArgumentParser(description='Arguments for training password strength detector.')

    args.add_argument('--t1', default=0.25, type=float, help='Train-valid/test split')
    args.add_argument('--t2', default=0.4, type=float, help='Train/valid split')

    args.add_argument('--iter', default=5000, type=int, help='Max iter for log reg')
    args.add_argument('--n_jobs', default=-1, type=int, help='Number of threads')
    args.add_argument('-C', type=float, default=1.0, help='Regularization factor')

    args.add_argument('--data_dir', type=str, default=r'data\dataset\password_dataset.csv',
                      help='Data directory')

    args.add_argument('--arch_dir', type=str, default=r'data\data archive\password_dataset.zip',
                      help='Compressed data directory')

    args.add_argument('--train', default=True, type=bool, choices=[True, False], help='Show train scores')
    args.add_argument('--valid', default=True, type=bool, choices=[True, False], help='Show valid scores')
    args.add_argument('--test', default=True, type=bool, choices=[True, False], help='Show test scores')

    args.add_argument('--matrix', default=True, type=bool, choices=[True, False], help='Show confusion matrix')

    args.add_argument('--acc', default=True, type=bool, choices=[True, False], help='Show accuracy score')
    args.add_argument('--rec', default=True, type=bool, choices=[True, False], help='Show recall score')
    args.add_argument('--pre', default=True, type=bool, choices=[True, False], help='Show precision score')
    args.add_argument('--f1', default=True, type=bool, choices=[True, False], help='Show f1 score')

    args.add_argument('--avg', default='macro', choices=['micro', 'macro', 'samples', 'weighted', 'binary'],
                      help='Metric aggregation')

    args.add_argument('--text', default=True, choices=[True, False], help='Display text with diagnostics')

    args.add_argument('--dp', default=7, type=int, help='Rounding precision for report metrics')

    args.add_argument('--save', default=True, type=bool, help='Save trained model')

    args.add_argument('--model_name', default='trained_model.pkl', type=str,
                      help='Name for trained model')

    args.add_argument('--model_dir', default=os.path.join(os.getcwd(), 'artefacts'),
                      type=str, help='Storage location for saved model')

    return args


def main():
    ### CLI arguments

    start_time = time.time()
    origin_time = deepcopy(start_time)

    print('>>> Parsing CLI arguments...')
    start_time = time.time()
    args = configure_args().parse_args()
    print(f'>>> CLI arguments parsed! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()

    ### Dataset
    print('>>> Importing dataset...')
    start_time = time.time()

    if not os.path.exists(args.arch_dir.replace('password_dataset.zip', '')):
        os.makedirs(args.arch_dir.replace('password_dataset.zip', ''))

    data, path_to_archive = read_compress(path_to_data = args.data_dir,
                                          path_to_archive = args.arch_dir)

    print(f'>>> Dataset successfully imported! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()

    ### Reproducibility
    print('>>> Ensuring reproducibility...')
    print('>>> Setting global and local random seeds...')

    start_time = time.time()

    random.seed(2022)
    os.environ['PYTHONHASHSEED'] = '2022'
    np.random.default_rng(2022)

    print('>>> Random seeds set!')
    print(f'>>> Reproducibility ensured! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()

    ### Feature generation
    print('>>> Generating features...')
    start_time = time.time()

    pattern = r'.*(qwert|qwer|rewq|wert|poiu|oiuy|bvcx|uytr|hgfd|iuyt|xcvb|sdfg|fghj|mnbv|jhgf|asdf|zxcv|poiuy|;lkj|lkjh|erty|rtyui|dfghj|cvbnm).*'
    new_data, targets = generate_features(data, pattern)

    print(f'>>> Features generated! Time elapsed : {time.time() - start_time:.5f} secs.',
          f'\n\tNumber of data observations : [{new_data.shape[0]}]',
          f'\n\tFeature dimensions : [{new_data.shape[1]}]')
    print()

    ### Encode target
    print('>>> Encoding target labels...')
    start_time = time.time()

    le = LabelEncoder()
    new_targets = le.fit_transform(targets)

    print(f'>>> Encoding completed! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()

    ### Data splitting
    print('>>> Splitting data into [train-valid-test] folds...')
    start_time = time.time()

    X_train, X_test, y_train, y_test = split_data(new_data, new_targets, split_size=args.t1)
    X_valid, X_test, y_valid, y_test = split_data(X_test, y_test, split_size=args.t2)

    print(f'>>> Data splits created! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()

    ### Model fitting
    model = import_model()
    model = model(max_iter=args.iter, n_jobs=args.n_jobs, C=args.C)

    print('>>> Training model...')
    model = train_model(model, X_train, y_train)
    print(f'>>> Model trained successfully! Time elapsed : {time.time() - start_time:.5f} secs.')
    print()

    if args.save:
        print('>>> Saving artefacts...')
        start_time = time.time()

        if not os.path.exists(args.model_dir):
            os.makedirs(os.path.join(args.model_dir, 'Model'))
            os.makedirs(os.path.join(os.getcwd(), 'data', 'generated data'), exist_ok=True)
        else:
            pass

        print('\t>>> Saving generated data...')
        new_data.to_csv(os.path.join(os.getcwd(), 'data', 'generated data', 'generated_features.csv'),
                        index=False)
        labels = pd.DataFrame(new_targets, columns=['strength'])

        labels.to_csv(os.path.join(os.getcwd(), 'data', 'generated data', 'generated_targets.csv'),
                      index=False)

        print('\t>>> Data saved successfully!')
        print()

        print('\t>>> Saving model artefacts...')

        with open(os.path.join(args.model_dir, 'Model', args.model_name), 'wb') as f:
            pickle.dump(model, f)

        print('\t>>> Model artefact saved!')
        print()

        if not os.path.exists(os.path.join(args.model_dir, 'Transformer')):
            os.makedirs(os.path.join(args.model_dir, 'Transformer'))

            with open(os.path.join(args.model_dir, 'Transformer', 'encoder.pkl'), 'wb') as f:
                pickle.dump(le, f)

            print('\t>>> Transformer artefact saved!')

        else:
            print('\t>>> Transformer artefact saved previously!')

        print(f'>>> Artefact redundancy achieved! Time elapsed : {time.time() - start_time:.5f} secs.')
        print()

    if args.matrix:
        if args.train:
            visualize_confusion_matrix(model, X_train, y_train)

        elif args.valid:
            visualize_confusion_matrix(model, X_valid, y_valid)

        elif args.test:
            visualize_confusion_matrix(model, X_test, y_test)

    if args.train:
        train_preds = model.predict(X_train)

        print('>' * 10, 'Train Diagnostics', '<' * 10)

        print(get_accuracy_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print(get_precision_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print(get_recall_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print(get_f1_score(y_train, train_preds, num_places=args.dp, text=args.text))
        print()

    if args.valid:
        valid_preds = model.predict(X_valid)

        print('>' * 10, 'Valid Diagnostics', '<' * 10)

        print(get_accuracy_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print(get_precision_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print(get_recall_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print(get_f1_score(y_valid, valid_preds, num_places=args.dp, text=args.text))
        print()

    if args.test:
        test_preds = model.predict(X_test)

        print('>' * 10, 'Test Diagnostics', '<' * 10)

        print(get_accuracy_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print(get_precision_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print(get_recall_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print(get_f1_score(y_test, test_preds, num_places=args.dp, text=args.text))
        print()

    print(f'>>> Program run successfully! Total Time elapsed :: {time.time() - origin_time} secs.')


if __name__ == '__main__':
    main()
