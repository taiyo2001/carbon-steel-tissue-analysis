print('exec')

import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
print('変更が反映されています')
