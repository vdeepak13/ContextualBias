import pickle
import time
import os
import torch
import numpy as np
import sys

from classifier import multilabel_classifier
from load_data import *

# Example use case:
# python stage1.py COCOStuff 100

modelpath = None # if we're training from scratch. Otherwise put the previous model checkpoint.

# Create data loader
dataset = sys.argv[1]
trainset = create_dataset(dataset, labels='labels_train.pkl', B=100) # instead of 200
valset = create_dataset(dataset, labels='labels_val.pkl', B=500)

nepochs = int(sys.argv[2])
print('Created train and val datasets \n')

if dataset == 'COCOStuff':
    unbiased_classes_mapped = pickle.load(open('{}/unbiased_classes_mapped.pkl'.format(dataset), 'rb'))

# Create output directory
outdir = '{}/save/stage1'.format(dataset)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
    print('Created', outdir, '\n')

# Initialize classifier
if dataset == 'COCOStuff':
    num_categs = 171
    learning_rate = 0.1
elif dataset == 'AwA':
    num_categs = 85
    learning_rate = 0.01
elif dataset == 'DeepFashion':
    num_categs = 250
    learning_rate = 0.1
else:
    num_categs = 0
    learning_rate = 0.01
    print('Invalid dataset: {}'.format(dataset))

Classifier = multilabel_classifier(torch.device('cuda'), torch.float32, 
                                   num_categs=num_categs, learning_rate=learning_rate, 
                                   modelpath=modelpath) # cuda

# Start stage 1 training
start_time = time.time()
for i in range(Classifier.epoch, nepochs):

    if dataset in ['COCOStuff', 'DeepFashion'] and i == 60: # Reduce learning rate from 0.1 to 0.01
        Classifier.optimizer = torch.optim.SGD(Classifier.model.parameters(), lr=0.01, momentum=0.9)

    Classifier.train(trainset)
    if (i + 1) % 5 == 0:
        Classifier.save_model('{}/stage1_{}.pth'.format(outdir, i))

    APs, mAP = Classifier.test(valset)
    if dataset == 'COCOStuff':
        mAP_unbiased = np.nanmean([APs[i] for i in unbiased_classes_mapped])
        print('Validation mAP: all {} {:.5f}, unbiased 60 {:.5f}'.format(num_categs, mAP, mAP_unbiased))
    else:
        print('Validation mAP: all {} {:.5f}'.format(num_categs, mAP))

    print('Time passed so far: {:.2f} minutes'.format((time.time()-start_time)/60.))
    print()
