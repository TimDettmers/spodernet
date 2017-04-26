import pytest
import torch
import time

from spodernet.utils.util import CUDATimer


@pytest.mark.torch
def test_cuda_timer():
    start = torch.cuda.Event(enable_timing=True, blocking=True)
    end = torch.cuda.Event(enable_timing=True, blocking=True)
    k = 100
    a = torch.rand(k,k).cuda()
    b = torch.rand(k,k).cuda()
    timer = CUDATimer()

    t0 = time.time()
    start.record()
    timer.tick('a')
    for i in range(1000):
        c = a*b
    end.record()
    timer_time = timer.tock('a')
    cpu_time = time.time() - t0
    cuda_time = start.elapsed_time(end)/1000.

    assert timer_time > cpu_time*0.9 and timer_time < cpu_time*1.1, 'CUDATimer imprecise with respect to CPU time.'
    assert timer_time > cuda_time*0.9 and timer_time < cuda_time*1.1, 'CUDATimer imprecise with respect to CUDA time.'
