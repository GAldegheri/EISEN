import torch

if __name__=="__main__":
    assert torch.cuda.is_available()
    print("All good!")
    print("Available devices:", torch._C._cuda_getDeviceCount())
    print("CUDA version:", torch.version.cuda)