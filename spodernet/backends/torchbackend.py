import torch
from torch.autograd import Variable

class TorchConverter(IAtBatchPreparedObservable):
    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts
        inp = Variable(torch.LongTensor(np.int64(inp)))
        inp_len = Variable(torch.IntTensor(np.int64(inp_len)))
        sup = Variable(torch.LongTensor(np.int64(sup)))
        sup_len = Variable(torch.IntTensor(np.int64(sup_len)))
        t = Variable(torch.LongTensor(np.int64(t)))
        return [inp, inp_len, sup, sup_len, t, idx]

class TorchCUDAConverter(IAtBatchPreparedObservable):
    def __init__(self, device_id):
        self.device_id = device_id

    def at_batch_prepared(self, batch_parts):
        inp, inp_len, sup, sup_len, t, idx = batch_parts
        inp = inp.cuda(self.device_id)
        inp_len = inp_len.cuda(self.device_id)
        sup = sup.cuda(self.device_id)
        sup_len = sup_len.cuda(self.device_id)
        t = t.cuda(self.device_id)
        idx = idx
        return [inp, inp_len, sup, sup_len, t, idx]


######################################
#
#          Util functions
#
######################################

def convert_state(state):
    if isinstance(state.targets, Variable):
        state.targets = state.targets.data
    if isinstance(state.argmax, Variable):
        state.argmax = state.argmax.data
    if isinstance(state.pred, Variable):
        state.pred = state.pred.data

    return state
