#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import LabelBinarizer
from Randomizer import LDPRandomizer as rz
from FL import FedMLFunc as fl
from FL import Net as Net
import torch, torch.nn as nn
from Helper import HelperFunctions as hf
import time
from tqdm import tqdm
rz = rz()
fl = fl()
hf = hf()





np.random.seed(25)
num_classes = 10 

# Client training settings
localepochs = 50 # The number of epochs for local model training. 50 is the default value    
weight_decay = 1e-4

# FL settings
the_highest_num_clients = 100 # The highest number of clients tested. 
num_of_clients = 2  # The number of clients that are currently running
if num_of_clients > 2:
    num_selected = num_of_clients - 1 # The number of clients randomly selected (participate) in each round of FL
else:
    num_selected = num_of_clients
num_rounds = 200 # 400 is the default value
epochs = 50 # The number of epochs for the clients during FL 
batch_size = 32





# The following data loading and normalization steps were 
# motivated by https://www.kaggle.com/code/dimitriosroussis/svhn-classification-with-cnn-keras-96-acc

# Load the data
train_raw = loadmat('./data/train_32x32.mat')
train_raw2 = loadmat('./data/extra_32x32.mat')
train_raw = dict(list(train_raw.items()) + list(train_raw2.items()))
test_raw = loadmat('./data/test_32x32.mat')

# Load images and labels
train_images = np.array(train_raw['X'])
test_images = np.array(test_raw['X'])
train_labels = train_raw['y']
test_labels = test_raw['y']

# Fix the axes of the images
train_images = np.moveaxis(train_images, -1, 0)
test_images = np.moveaxis(test_images, -1, 0)

print(len(test_images))
print(len(train_images))
test_images = test_images[:5000]
train_images = train_images[:50000]
# Convert train and test images into 'float64' type
train_images = train_images.astype('float64')
test_images = test_images.astype('float64')

# Convert train and test labels into 'int64' type
tr_labels = train_labels.astype('int64')
te_labels = test_labels.astype('int64')

# Normalize the image data
train_images /= 255.0
test_images /= 255.0







# One-hot encoding of train and test labels
lb = LabelBinarizer()
train_labels = lb.fit_transform(tr_labels)
test_labels = lb.fit_transform(te_labels)

# Delare and initialize additional variables on training and testing data
Y_train = train_labels 
Y_test = test_labels

x_train_full = train_images
y_train_full = train_labels
x_test_full = test_images
y_test_full = test_labels

# Calculate the number of tuples that should be taken from the input dataset for training and testing
n_tup_client_tr = int(x_train_full.shape[0]/the_highest_num_clients)
tot_used_for_training = num_of_clients * n_tup_client_tr

n_tup_client_ts = int(x_test_full.shape[0]/the_highest_num_clients)
tot_used_for_testing = num_of_clients * n_tup_client_ts

# Trimming the full training and testing datasets based on the number of clients
x_train_full = train_images[0:tot_used_for_training]
y_train_full = Y_train[0:tot_used_for_training]
x_test_full = test_images[0:tot_used_for_testing]
y_test_full = Y_test[0:tot_used_for_testing]
 
# Generating training data partitions for training the model and generating randomized data
x_train_partitions = np.array_split(x_train_full, num_of_clients)
y_train_partitions = np.array_split(y_train_full, num_of_clients)

# Generating the testing data splits for the randomization of testing data
x_test_partitions = np.array_split(x_test_full, num_of_clients)





# Checking the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Getting GPU usage information before computation
if device.type == 'cuda':
    for i in range(0,torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
        print('Memory Usage of device :', i)
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        
        
        
        
r_x_train = []
r_x_test = []

for i in tqdm(range(0,num_of_clients)):
    client_inter_model = hf.client_model_create(i, x_train_partitions[i],y_train_partitions[i],weight_decay, num_classes, batch_size,localepochs)
    print("######## predicting x_train ###########")
    x_train_flat = client_inter_model.predict(x_train_partitions[i]) # Obtaining the flattend vectors for the training data   
    print("######## randomizing x_train ###########")
    r_x_train.append(rz.flattenrand(x_train_flat)) # Randomizing the training data
    print("############# predicting x_test ##############")
    x_test_flat = client_inter_model.predict(x_test_partitions[i]) # Obtaining the flattend vectors for the testing data
    print("############ randomizing x_test #############")
    r_x_test.append(np.array(rz.flattenrand(x_test_flat))) # Randomizing the testing data
    
    
    
    
torch.cuda.empty_cache()
# Instantiate models and optimizers
global_model = Net()
client_models = [Net() for _ in range(num_of_clients)]
for model in client_models:
    model.load_state_dict(global_model.state_dict())

opt = [optim.SGD(model.parameters(), lr=0.001) for model in client_models]

tensor_x_train = []
tensor_y_train = []

for i in range(0,num_of_clients):
    tensor_x_train.append((torch.tensor(r_x_train[i])).type(torch.FloatTensor))   
    y_train_reverse_onehot = np.array( [ np.argmax ( y, axis=None, out=None ) for y in y_train_partitions[i] ] )
    tensor_y_train.append((torch.tensor(y_train_reverse_onehot)).type(torch.LongTensor))

r_x_test_array = np.vstack((r_x_test))

y_test_reverse_onehot = np.array( [ np.argmax ( y, axis=None, out=None ) for y in y_test_full ] )
tensor_x_test = (torch.tensor(r_x_test_array)).type(torch.FloatTensor)
tensor_y_test = (torch.tensor(y_test_reverse_onehot)).type(torch.LongTensor)
dataloaders_test = torch.utils.data.DataLoader(tensor_x_test, batch_size=64)
dataloaders_labels_test= torch.utils.data.DataLoader(tensor_y_test, batch_size=64)





acc_train_collect = []
acc_test_collect = []
loss_train_collect = []
loss_test_collect = []

for r in tqdm(range(num_rounds)):
    # select (num_of_clients - 1) clients randomly
    client_idx = np.random.permutation(num_of_clients)[:num_selected]
    trainloss = 0
    trainacc = 0
    # client update
    loss = 0
    for i in client_idx:  
        dataloaders_train = torch.utils.data.DataLoader(tensor_x_train[i], batch_size=64)
        dataloaders_labels= torch.utils.data.DataLoader(tensor_y_train[i], batch_size=64)  
        [loss,acc]= fl.client_update(client_models[i], opt[i], zip(dataloaders_train, dataloaders_labels), epoch=epochs)    
        
        trainloss += loss
        trainacc += acc
    # server aggregate
    cm = np.asarray(client_models)
    fl.server_aggregate(global_model, list(cm[tuple([client_idx])]))
    test_loss, test_acc = fl.test(global_model, zip(dataloaders_test, dataloaders_labels_test),dataloaders_test)
    
    print('loss %0.3g - accuracy: %0.3g  - test_loss %0.3g - test_accuracy: %0.3f' % (trainloss / len(client_idx), trainacc / len(client_idx), test_loss, test_acc))
    
    acc_train_collect.append(trainacc / len(client_idx))
    acc_test_collect.append(test_acc)
    loss_train_collect.append(trainloss / len(client_idx))
    loss_test_collect.append(test_loss)

print("Training and Evaluation completed!") 





# Plotting loss
f1 = plt.figure()
plt.plot(np.arange(1, num_rounds+1), loss_train_collect, label='Train', linestyle = 'dashed')
plt.plot(np.arange(1, num_rounds+1), loss_test_collect, label='Test', linestyle = 'solid')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend()

# Plotting accuracy
f2 = plt.figure()
plt.plot(np.arange(1, num_rounds+1), acc_train_collect, label='Train', linestyle = 'dashed')
plt.plot(np.arange(1, num_rounds+1), acc_test_collect, label='Test', linestyle = 'solid')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.legend()


# GPU usage information after computation
if device.type == 'cuda':
    for i in range(0,torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
        print('Memory Usage of device :', i)
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB') 



