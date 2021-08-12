import torch
import time
print(f" before to_cpu\t"
        f"Memory: {float(torch.cuda.memory_allocated(device=1)):.3f} ({float(torch.cuda.memory_cached(device=1)):.3f})")
a=torch.ones([320, 64, 224, 224],device=1)
print(f" before to_cpu\t"
        f"Memory: {float(torch.cuda.memory_allocated(device=1)):.3f} ({float(torch.cuda.memory_cached(device=1)):.3f})")
time.sleep(1)
# print(a)
if a.is_cuda:
    a=a.cpu()
    a=None
    print(f" after to_cpu\t"
        f"Memory: {float(torch.cuda.memory_allocated(device=1)):.3f} ({float(torch.cuda.memory_cached(device=1)):.3f})")
     
