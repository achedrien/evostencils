import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


dataset_root = "./data/train/"
shape = "curve"
image_size = 128
start_index = 1
num_instances = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in range(start_index, start_index + num_instances):
    x = np.zeros((image_size, image_size), dtype=np.float128)
    middle = np.random.rand(image_size-2, image_size-2)
    x[1:-1, 1:-1] = middle
    x = torch.from_numpy(x[np.newaxis, np.newaxis, :, :].astype(np.float64)).to(device)
    fixed_stencil = torch.tensor([[0.0, 1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0, 1.0, 0.0]], dtype=torch.float64).to(device).unsqueeze(0).unsqueeze(0)
    conv_holder = F.conv2d(x, (1 / (1 / np.shape(x)[-1])**2) * fixed_stencil, padding=0)
    b = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
    print(x.shape)
    data: np.ndarray = np.stack([x, b])
    filename = os.path.join(dataset_root, 'data_{:06d}'.format(i))
    np.save(filename, data)
