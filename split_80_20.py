import pickle
import numpy as np
import sys

dataset = sys.argv[1]

# Load the processed train labels
labels_train = pickle.load(open('/n/fs/context-scr/{}/labels_train.pkl'.format(dataset), 'rb'))

# Do a 80-20 split of train
N = len(list(labels_train.keys()))
N_80 = int(N*0.8)

# Select images to be added to the 80 split
np.random.seed(1234)
keys = list(labels_train.keys())
inds_80 = np.random.choice(N, N_80, replace=False)
keys_80 = np.array(keys)[inds_80]
keys_20 = np.delete(keys, inds_80)

# Create smaller label dictionaries
labels_train_80 = {k: labels_train[k] for k in keys_80}
labels_train_20 = {k: labels_train[k] for k in keys_20}

with open('/n/fs/context-scr/{}/labels_train_80.pkl'.format(dataset), 'wb') as handle:
    pickle.dump(labels_train_80, handle, protocol=4)
with open('/n/fs/context-scr/{}/labels_train_20.pkl'.format(dataset), 'wb') as handle:
    pickle.dump(labels_train_20, handle, protocol=4)
