import numpy as np

mfcc = np.load('data/features/mfcc.npy')
mfcc_labels = np.load('data/features/mfcc_labels.npy')
print(mfcc.shape)        # (N, 128, 16) байх ёстой
print(mfcc_labels.shape) # (N,)