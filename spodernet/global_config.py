from collections import namedtuple

from spodernet.logger import Logger
log = Logger('global_config.py.txt')

class Backends:
    TORCH = 'pytorch'
    TENSORFLOW = 'tensorflow'
    NUMPY = 'numpy'


class Config:
    dropout = 0.0
    batch_size = 128
    learning_rate = 0.001
    backend = Backends.TORCH

    @staticmethod
    def parse_argv(*argv):
        file_name = argv[0]
        args = argv[1:]
        assert len(args) % 2 == 0, 'Global parser expects an even number of arguments.'
        values = []
        names = []
        for i, token in enumerate(args):
            if i % 2 == 0:
                names.append(token)
            else:
                values.append(token)

        for i in range(len(names)):
            if names[i] in alias2params:
                log.debug('Replaced parameters alias {0} with name {1}', name[i], alias2params[name[i]])
                names[i] = alias2params[names[i]]

        for i in range(len(names)):
            name = names[i]
            if name[:2] == '--': continue
            assert name in params2type, 'Parameter {0} does not exist. Prefix your custom parameters with -- to skip parsing for global config'.format(name)
            values[i] = params2type[name](values[i])

        for name, value in zip(names, values):
            if name[:2] == '--': continue
            params2field[name](value)
            log.debug('Set parameter {0} to {1}', name, value)


params2type = {}
params2type['learning_rate'] = lambda x: float(x)
params2type['dropout'] = lambda x: float(x)
params2type['batch_size'] = lambda x: int(x)

alias2params = {}
alias2params['lr'] = 'learning_rate'


params2field = {}
params2field['learning_rate'] = lambda x: setattr(Config, 'learning_rate', x)
params2field['dropout'] = lambda x: setattr(Config, 'dropout', x)
params2field['batch_size'] = lambda x: setattr(Config, 'batch_size', x)


