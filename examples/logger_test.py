from spodernet.logger import Logger, LogLevel

Logger.GLOBAL_LOG_LEVEL = LogLevel.DEBUG
l = Logger('test', 'logger_test.py')


l.debug('abc {0}, {1}, {2}', '234', 2434, 'aaaa')



Logger.GLOBAL_LOG_LEVEL = LogLevel.STATISTICAL
Logger.LOG_PROPABILITY = 0.001
for i in range(10000):
    l.statistical('this is a int: {0}', i)
