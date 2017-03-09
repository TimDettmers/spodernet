'''Downloads SNLI data and wrangles it into the spoder format'''
from __future__ import print_function
import os
from os.path import join
import urllib
import zipfile
import simplejson as json


def download_snli():
    '''Creates data and snli paths and downloads SNLI in the home dir'''
    home = os.environ['HOME']
    data_dir = join(home, '.data')
    snli_dir = join(data_dir, 'snli')
    snli_url = 'http://nlp.stanford.edu/projects/snli/snli_1.0.zip'

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(snli_dir):
        os.mkdir(snli_dir)

    if not os.path.exists(join(data_dir, 'snli_1.0.zip')):
        print('Downloading SNLI...')
        snlidownload = urllib.URLopener()
        snlidownload.retrieve(snli_url, join(data_dir, "snli_1.0.zip"))

    print('Opening zip file...')
    archive = zipfile.ZipFile(join(data_dir, 'snli_1.0.zip'), 'r')

    return archive, snli_dir


def snli2spoder():
    '''Preprocesses SNLI data and returns to spoder files'''
    files = ['snli_1.0_train.jsonl', 'snli_1.0_dev.jsonl',
             'snli_1.0_test.jsonl']

    archive, snli_dir = download_snli()

    new_files = ['train.data', 'dev.data', 'test.data']
    names = ['train', 'dev', 'test']

    if not os.path.exists(join(snli_dir, new_files[0])):
        for name, new_name in zip(files, new_files):
            print('Writing {0}...'.format(new_name))
            snli_file = archive.open(join('snli_1.0', name), 'r')
            with open(join(snli_dir, new_name), 'wb') as datafile:
                for line in snli_file:
                    data = json.loads((line))
                    if data['gold_label'] == '-':
                        continue

                    premise = data['sentence1']
                    hypothesis = data['sentence2']
                    target = data['gold_label']
                    datafile.write(
                        json.dumps([premise, hypothesis, target]) + '\n')

    return [names, [join(snli_dir, new_name) for new_name in new_files]]

if __name__ == '__main__':
    snli2spoder()
