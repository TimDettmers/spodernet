from __future__ import print_function
import pytest

from spodernet.utils.logger import Logger, GlobalLogger



def test_global_logger():
    log1 = Logger('test1.txt')
    log2 = Logger('test2.txt')
    log1.info('uden')
    log2.info('kek')
    log2.info('rolfen')
    log1.info('keken')


    GlobalLogger.flush()

    expected = ['uden', 'kek', 'rolfen', 'keken']
    with open(GlobalLogger.global_logger_path) as f:
        data = f.readlines()



    print(len(data))
    for i, line in enumerate(data[-4:]):
        message = line.split(':')[3].strip()
        assert message == expected[i]

    assert i  == len(expected) - 1
