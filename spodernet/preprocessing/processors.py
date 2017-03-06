from spodernet.logger import Logger

log = Logger('processors.py.txt')

class AbstractProcessor(object):
    def __init__(self):
        self.state = None
        pass


    def link_with_pipeline(self, state):
        self.state = state

    def process(self, inputs, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractProcessor need to implement the process method')

class AbstractLoopLevelTokenProcessor(AbstractProcessor):
    def __init__(self, successive_for_loops_to_tokens=2):
        super(AbstractLoopLevelTokenProcessor, self).__init__()
        self.successive_for_loops = successive_for_loops_to_tokens

    def process_token(self, token, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractLoopLevelTokenProcessor need to implement the process_token method ')

    def process(self, sample, inp_type):
        if self.successive_for_loops == 0:
            ret = self.process_token(sample, inp_type)

        elif self.successive_for_loops == 1:
            new_tokens = []
            for token in sample:
                new_tokens.append(self.process_token(token, inp_type))
            ret = new_tokens

        elif self.successive_for_loops == 2:
            new_sents = []
            for sent in sample:
                new_tokens = []
                for token in sent:
                    new_tokens.append(self.process_token(token, inp_type))
                new_sents.append(new_tokens)
            ret = new_sents

        return ret

class AbstractLoopLevelListOfTokensProcessor(AbstractProcessor):
    def __init__(self, successive_for_loops_to_tokens=1):
        super(AbstractLoopLevelListOfTokensProcessor, self).__init__()
        self.successive_for_loops = successive_for_loops_to_tokens

    def process_list_of_tokens(self, tokens, inp_type):
        raise NotImplementedError('Classes that inherit from AbstractLoopLevelListOfTokensProcessor need to implement the process_list_of_tokens method ')

    def process(self, sample, inp_type):
        if self.successive_for_loops == 0:
            ret = self.process_list_of_tokens(sample, inp_type)

        elif self.successive_for_loops == 1:
            new_sents = []
            for sent in sample:
                new_sents.append(self.process_list_of_tokens(sent, inp_type))
            ret = new_sents

        return ret

class Tokenizer(AbstractProcessor):
    def __init__(self, tokenizer_method):
        super(Tokenizer, self).__init__()
        self.tokenize = tokenizer_method

    def process(self, sentence, inp_type):
        return self.tokenize(sentence)

class AddToVocab(AbstractProcessor):
    def __init__(self):
        super(AddToVocab, self).__init__()

    def process(self, token, inp_type):
        if inp_type == 'target':
            self.state['vocab'].add_label(token)
        else:
            self.state['vocab'].add_token(token)
        return token

class ToLower(AbstractProcessor):
    def __init__(self):
        super(ToLower, self).__init__()

    def process(self, token, inp_type):
        return token.lower()

class ConvertTokenToIdx(AbstractLoopLevelTokenProcessor):
    def __init__(self, successive_for_loops_to_tokens=2, is_label=False):
        super(ConvertTokenToIdx, self).__init__(successive_for_loops_to_tokens)
        self.successive_for_loops = successive_for_loops_to_tokens

    def process_token(self, token, inp_type):
        if inp_type != 'target':
            log.statistical('a label {0}', token)
            return self.state['vocab'].get_idx(token)
        else:
            log.statistical('a non-label token {0}', token)
            return self.state['vocab'].get_idx_label(token)

class SaveStateToList(AbstractProcessor):
    def __init__(self, name):
        super(SaveStateToList, self).__init__()
        self.name = name

    def link_with_pipeline(self, state):
        self.state = state
        if self.name not in self.state['data']:
            self.state['data'][self.name] = {}
        self.data = self.state['data'][self.name]

    def process(self, data, inp_type):
        if inp_type not in self.data: self.data[inp_type] = []
        self.data[inp_type].append(data)
        return data

class SaveLengths(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, successive_for_loops_to_sentences=1):
        super(SaveLengths, self).__init__(successive_for_loops_to_sentences)

    def link_with_pipeline(self, state):
        self.state = state
        if 'lengths' not in self.state['data']:
            self.state['data']['lengths'] = {}
        self.data = self.state['data']['lengths']

    def process_list_of_tokens(self, tokens, inp_type):
        if inp_type not in self.data: self.data[inp_type] = []
        self.data[inp_type].append(len(tokens))
        log.statistical('A list of tokens: {0}', tokens)
        return tokens

class StreamToHDF5(AbstractLoopLevelListOfTokensProcessor):
    def __init__(self, successive_for_loops_to_sentences=1, samples_per_file=50000, max_length=None):
        super(StreamToHDF5, self).__init__(successive_for_loops_to_sentences)
        self.max_length = max_length
        self.samples_per_file = samples_per_file

    def link_with_pipeline(self, state):
        self.state = state
        self.shard_id = 1
        self.root = self.state['path']
        self.data = {'target' : [], 'input' : [], 'support' : []}

    def process_list_of_tokens(self, tokens, inp_type):
        if inp_type != 'target':
            if self.max_length is None:
                if 'length' not in self.state['data']:
                    log.error('Either specify the max_length or create lengths via the "SaveLengths" processor')
                if inp_type not in self.state['data']['length']:
                    log.error('Either specify the max_length or create lengths via the "SaveLengths" processor')
                max_length = np.max(self.state['data']['length'][inp_type])
                log.statistical('max length of the dataset: {0}', max_length)
                log.debug('using type int32 for inputs and supports for now, but this may not be correct in the future') 
                x = np.zeros((1,max_length), dtype=np.int32)
                x[:len(tokens)] = tokens
                self.data[inp_type].append(x)
        else:
            data['target'].append(tokens)

        if len(data) == self.samples_per_file:
            pass

        return tokens

    def save_to_hdf5(self):
        self.data = {'target' : [], 'input' : [], 'support' : []}
        self.shard_id += 1
        pass
