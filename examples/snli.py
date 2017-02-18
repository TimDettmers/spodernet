'''This models is an example for training a classifier on SNLI'''
from __future__ import print_function

from spodernet.data.snli2spoder import snli2spoder
from spodernet.preprocessing import spoder2hdf5
from spodernet.util import hdf2numpy


def main():
    names, file_paths = snli2spoder()
    lower_list = [True for name in names]
    add_to_vocab_list = [name != 'test' for name in names]
    filetype = spoder2hdf5.SINGLE_INPUT_SINGLE_SUPPORT_CLASSIFICATION
    hdf5paths, vocab = spoder2hdf5.file2hdf(
        file_paths, names, lower_list, add_to_vocab_list, filetype)

    datasets = []
    for path in hdf5paths:
        I, S = path
        X = hdf2numpy(I)
        for row in X:
            sent = ''
            for num in row:
                sent += vocab.get_word(num) + ' '
            print(sent)
        datasets.append(hdf2numpy(I))


if __name__ == '__main__':
    main()
