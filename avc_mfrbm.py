import torch
import torchvision
import numpy as np
from load_avc import create_df
import learnergy.visual.tensor as t
import os
#from learnergy.models.deep import DBN
from learnergy.models.gaussian import GaussianRBM
#from learnergy.models.gaussian.mult_frbm import MultFRBM#FRRBM
from core.mult_frbm import MultFRBM#FRRBM

# Creating training and testing dataset
dy, dx = 50, 50
rep = 15
split = 0.75
# phase; hid=500; gauss=1500; lr = 0.001; momentum=0.9 -> better result
# 0 = mag+phase+gauss; 
# 1 = mag+gauss; 
# 2 = phase+gauss

for r in range(rep):
    np.random.seed(r)
    
    train, test = create_df(os.getcwd(), dy, dx, '*.jpg', split=split)

    # Creating a MultFRRBM
    model = MultFRBM(model=['fourier', 'gaussian'], n_visible=(dy,dx), n_hidden=(500,1500), 
			steps=(1, 1), learning_rate=(0.001, 0.0001),
			momentum=(0.9, 0.5), decay=(0,0), temperature=(1,1), use_gpu=False,
			fr_features=['phase'])

    #model = GaussianRBM(n_visible=dy*dx, n_hidden=2000, steps=1, learning_rate=0.0001,
    #                 momentum=0.5, decay=0, temperature=1, use_gpu=False)
    #model = DBN(model=['mult_fourier'], n_visible=(dy,dx), n_hidden=(2000, 2000), steps=(1, 1), 
    #            learning_rate=(0.0001, 0.0001), momentum=(0.5, 0.5), decay=(0,0), temperature=(1, 1), use_gpu=False)

    if str(model) == "MultFRBM()":
        name = '2multfrrbm'+str(r)+'.pth'
    else:
        name = 'rbm'+str(r)+'.pth'
    print("", name, r)

    # Training a MultFRBM
    mse, pl = model.fit(train, batch_size=10, epochs=50)
    #mse, pl = model.fit(train, batch_size=10, epochs=(40, 40))

    # Saving model
    #torch.save(model, 'multfrrbm'+str(r)+'.pth')
    torch.save(model, name)

    import classification_avc
    from classification_avc import classification
    classification(name=name, seed=r, split=split)

