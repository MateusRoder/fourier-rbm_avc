import os
import numpy as np
import glob
import torch
from learnergy.core.dataset import Dataset
from PIL import Image

def create_df(path, dy=50, dx=50, Type='*.jpg', split=0.8):
    #np.random.seed(0)
    root = path
    pathA = path + str("/avc/Hemorragico")
    pathB = path + str("/avc/Isquemico")
    pathC = path + str("/avc/Normal")
    path = pathA, pathB, pathC
    data = []
    cont = 0
    label = []
    for i in range(3):
        os.chdir(path[i])
        files = glob.glob(Type)
        for j in range(len(files)):
            img = Image.open(files[j]).convert('L')
            img = img.resize((dy, dx))
            img = np.array(img)
            img = img[:,:]/255.  ## gray scaled ##
            data.append(img)
            #data.append(np.ndarray.flatten(img))
            label.append(i)
            cont+=1
        #size = len(np.ndarray.flatten(img))

    df2 = (np.reshape(data, ((cont, dy, dx))))
    lb = (np.reshape(np.array(label), ((cont,1))))
    df = np.concatenate((df2.reshape((len(df2), dy*dx)), lb), axis=-1)

    np.random.shuffle(df)
    lb = np.array(df[:,dy*dx:].reshape((len(lb))), 'int32')
    df = np.array(df[:,:dy*dx].reshape((len(df), dy, dx)), 'float32')
    perc = int((1-split)*len(df))
    total = len(df) - perc

    train = Dataset(data=df[perc:], targets=lb[perc:], transform=None)
    test = Dataset(data=df[:perc], targets=lb[:perc], transform=None)

    os.chdir(root)
    print("Dataset Loaded!")
    return train, test
