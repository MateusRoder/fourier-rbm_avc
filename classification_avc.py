import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
#import learnergy.visual.tensor as t
#import matplotlib.pyplot as plt
import numpy as np
import os

from torch.utils.data import DataLoader
from tqdm import tqdm
from load_avc import create_df
from sklearn.metrics import confusion_matrix#, plot_confusion_matrix

def classification(name, seed, split=0.75):
    np.random.seed(seed)
    # Creating training and testing dataset
    dy, dx = 50, 50
    train, test = create_df(os.getcwd(), dy, dx, '*.jpg', split=split)

    n_classes = 3
    batch_size = 60
    fine_tune_epochs = 50

    model = torch.load(name)
    model.eval()
    tam = len(model.models)

    # Creating the Fully Connected layer to append on top of RBM
    try:
        #if len(model.models)>=2:
        hid = model.n_hidden[0]
        for j in range(1, tam):
            hid+=model.n_hidden[j]
        fc = nn.Linear(int(hid), n_classes)
    except:
        try:
            fc = nn.Linear(model.models[model.n_layers-1].n_hidden, n_classes)
        except:
            fc = nn.Linear(model.n_hidden, n_classes)

    # Check if model uses GPU
    if model.device == 'cuda':
        # If yes, put fully-connected on GPU
        fc = fc.cuda()

    # Cross-Entropy loss is used for the discriminative fine-tuning
    criterion = nn.CrossEntropyLoss()

    # Creating the optimzers
    try:
        optimizer = []
        for j in range(tam):
            optimizer.append(optim.Adam(model.models[j].parameters(), lr=0.00001))
        optimizer.append(optim.Adam(fc.parameters(), lr=0.001))
        #optimizer = [optim.Adam(model.models[0].parameters(), lr=0.00001),
        #            optim.Adam(model.models[1].parameters(), lr=0.00001),
        #            optim.Adam(fc.parameters(), lr=0.001)]
        #optimizer = [optim.Adam(fc.parameters(), lr=0.0001)]
    except:
        optimizer = [optim.Adam(model.parameters(), lr=0.00001),
                    optim.Adam(fc.parameters(), lr=0.001)]

    # Creating training and validation batches
    train_batch = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_batch = DataLoader(test, batch_size=len(test.data), shuffle=False, num_workers=0)
    save_acc = np.zeros((fine_tune_epochs))
    # For amount of fine-tuning epochs
    for e in range(fine_tune_epochs):
        print(f'Epoch {e+1}/{fine_tune_epochs}')

        # Resetting metrics
        train_loss, val_acc = 0, 0
    
        # For every possible batch
        for x_batch, y_batch in tqdm(train_batch):

            # For every possible optimizer
            for opt in optimizer:
                # Resets the optimizer
                opt.zero_grad()

            # Checking whether GPU is avaliable and if it should be used
            if model.device == 'cuda':
                # Applies the GPU usage to the data and labels
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            y_batch = torch.tensor(y_batch.detach(), dtype=torch.long)

            # Passing the batch down the model
            #print("x", x_batch.shape)
            y = model(x_batch)

            # Calculating the fully-connected outputs
            y = fc(y)

            # Calculating loss
            loss = criterion(y, y_batch)
        
            # Propagating the loss to calculate the gradients
            loss.backward()
        
            # For every possible optimizer
            for opt in optimizer:
                # Performs the gradient update
                opt.step()

            # Adding current batch loss
            train_loss += loss.item()
        
        # Calculate the test accuracy for the model:
        for x_batch, y_batch in tqdm(val_batch):

            # Checking whether GPU is avaliable and if it should be used
            if model.device == 'cuda':
                # Applies the GPU usage to the data and labels
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            y_batch = torch.tensor(y_batch.detach(), dtype=torch.long)
 
            # Passing the batch down the model
            y = model(x_batch)

            # Calculating the fully-connected outputs
            y = fc(y)

            # Calculating predictions
            _, preds = torch.max(y, 1)

            # Calculating validation set accuracy
            val_acc = torch.mean((torch.sum(preds == y_batch).float()) / preds.size(0))

        print(f'Loss: {train_loss / len(train_batch)} | Val Accuracy: {val_acc}')
        save_acc[e] = val_acc

    cm = np.array(confusion_matrix(y_true=y_batch, y_pred=preds, labels=[0,1,2]), 'int32')
    np.savetxt('02conf_matrix'+str(seed)+'.txt', cm)
    np.savetxt('02test_acc'+str(seed)+'.txt', save_acc)
    #plt.plot(save_acc)
    #plt.show()
    # Saving the fine-tuned model
    #torch.save(model, 'tuned_'+name)

#for i in range(15):
#    classification('0multfrrbm'+str(i)+'.pth', i, 0.75)

