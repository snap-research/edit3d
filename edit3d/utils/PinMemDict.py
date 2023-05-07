import torch

# czz: the pinned memory is just used to speed up the transfer between CPU and GPU
# because the data is loaded on CPU first.
class PinMemDict:
    def __init__(self, data):
        self.data = data

    # custom memory pinning method on custom type
    def pin_memory(self):
        out_b = {}
        for k, v in self.data.items():
            if torch.is_tensor(v):
                out_b[k] = v.pin_memory()
            else:
                out_b[k] = v
        return out_b
