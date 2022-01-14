''' Make predictions via trained model. '''

import pickle
import os

from model_utils import get_predictions


def predictions():
    ### Setup options

    option = input('> Please input prediction type [batch OR online]: ')
    print()

    while option.lower().strip() not in ['batch', 'online']:
        print('Option must be either batch or online')
        option = input('> Please input prediction type [batch OR online]: ')
        print()

    if option == 'online':
        multi = input('> Please input single case-sensitive password: ')
        print()
        final = bool(int(input('> Return generated features? [1 for Yes; 0 for No]: ')))
    else:
        multi = []
        single_pass = input('> Please input case-sensitive password. Input -1 to proceed: ')

        while single_pass != str(-1):
            multi.append(single_pass)
            single_pass = input('> Please input case-sensitive password. Input -1 to proceed: ')

        print()
        final = bool(int(input('> Return generated features? [1 for yes; 0 for No]: ')))

    print()

    ### Load artefacts
    print('>>> Loading artefacts...')

    print('\t>>> Loading model...')
    with open('artefacts/Model/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print('\t>>> Model successfully loaded!')
    print()

    print('\t>>> Loading encoder...')
    with open('artefacts/Transformer/encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print('\t>>> Encoder successfully loaded!')
    print()

    results = get_predictions(passwords = multi, model=model, transformer=le, return_features=final)

    if option == 'online':
        print(f'\t\tPassword `{results[0].item()}` is predicted to be: {results[1].item().title()}.')
    else:
        for p, r in zip(results[0], results[1]):
            print(f'\t\tPassword `{p}` is predicted to be: {r.title()}.')

    print()

    if final:
        save_final = bool(int(input('> Save generated features? [1 for yes; 0 for No]: ')))
        print()

        if save_final:
            if not os.path.exists(os.path.join('data', 'generated predictions')):
                os.makedirs(os.path.join('data', 'generated predictions'))

            print('>>> Saving predictions...')
            results[-1].to_csv('data/generated predictions' +
                               input('> Please input save name: ') + '.csv', index = False)

            print('>>> Features saved!')


if __name__ == '__main__':
    predictions()
