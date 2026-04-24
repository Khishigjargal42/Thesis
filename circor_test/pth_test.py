import torch
state = torch.load("parallelcnn_mfcc_attention.pth", map_location="cpu")
for k, v in state.items():
    print(k, v.shape)