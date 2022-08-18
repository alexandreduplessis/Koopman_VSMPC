import json
import numpy as np
import torch

a = torch.tensor([1,2,3,4,5])
dict_torch  = {}
dict_torch['a'] = a
b = np.array([1,2,3,4,5])
dict_np = {}
dict_np['b'] = b
# with open('test_torch.json', 'w') as f:
#     json.dump(dict_torch, f)
torch.save(dict_torch, 'tensors_backup/test_torch.pt')