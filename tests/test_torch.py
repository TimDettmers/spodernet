import pytest
import torch
import time

from spodernet.utils.util import CUDATimer


@pytest.mark.torch
def test_cuda_timer():
    start = torch.cuda.Event(enable_timing=True, blocking=True)
    end = torch.cuda.Event(enable_timing=True, blocking=True)
    k = 1000
    a = torch.rand(k,k).cuda()
    b = torch.rand(k,k).cuda()
    timer = CUDATimer()

    t0 = time.time()
    start.record()
    timer.tick('a')
    timer.tick('timers')
    for i in range(10000):
        timer.tick('b')
        timer.tick('timers')
        c = a*b
        timer.tick('timers')
        timer.tick('b')
    torch.cuda.synchronize()
    end.record()
    overhead = timer.tock('timers')
    timer_time = timer.tock('a')
    timer_time_cumulative = timer.tock('b') + overhead
    cpu_time = time.time() - t0
    cuda_time = (start.elapsed_time(end)/1000.)

    print(cpu_time, cuda_time, timer_time, timer_time_cumulative)

    assert timer_time > cpu_time*0.9 and timer_time < cpu_time*1.1, 'CUDATimer imprecise with respect to CPU time.'
    assert timer_time > cuda_time*0.9 and timer_time < cuda_time*1.1, 'CUDATimer imprecise with respect to CUDA time.'
    assert timer_time_cumulative > cuda_time*0.9 and timer_time_cumulative < cuda_time*1.1, 'Cumulative CUDATimer imprecise with respect to CUDA time.'
