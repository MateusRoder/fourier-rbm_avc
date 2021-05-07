"""Fourier-based Multimodal Gaussian-Bernoulli Restricted Boltzmann Machine.
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader

from tqdm import tqdm

import learnergy.utils.constants as c
import learnergy.utils.exception as e
import learnergy.utils.logging as l
from learnergy.core import Model
#from learnergy.models.gaussian.fourier_rbm import FRRBM
from learnergy.models.gaussian.gaussian_rbm import GaussianRBM
from learnergy.models.extra.sigmoid_rbm import SigmoidRBM

from core.fourier_rbm import FRRBM
from torch.fft import fftn, ifftn
#from learnergy.utils.fft_utils import fftshift, ifftshift

logger = l.get_logger(__name__)


class MultFRBM(Model):
    """A Fourier-based RBM class provides the basic implementation for
    Gaussian-Bernoulli Restricted Boltzmann Machines (with standardization).

    This is a trick to ease the calculations of the hidden and
    visible layer samplings, as well as the cost function.

    References:
        Roder M, de Rosa G. H.
	...(2021)

    """

    def __init__(self, model=['fourier', 'fourier', 'gaussian'], n_visible=(128, 128, 128), n_hidden=(128, 128, 128), 			steps=(1, 1, 1), learning_rate=(0.001, 0.001, 0.001),
                momentum=(0, 0, 0), decay=(0, 0, 0), temperature=(1, 1, 1), use_gpu=False, 
		fr_features=['magnitude', 'phase']):
        """Initialization method.

        Args:
            n_visible (int): Amount of visible units.
            n_hidden (int): Amount of hidden units.
            steps (int): Number of Gibbs' sampling steps.
            learning_rate (float): Learning rate.
            momentum (float): Momentum parameter.
            decay (float): Weight decay used for penalization.
            temperature (float): Temperature factor.
            use_gpu (boolean): Whether GPU should be used or not.

        """

        logger.info('Overriding class: FRRBM -> MultFRBM.')
        # Override its parent class
        super(MultFRBM, self).__init__(use_gpu=use_gpu)

        # Shape of visible units
        self.visible_shape = n_visible

        # Amount of visible units
        self.n_visible = int(n_visible[0]*n_visible[1])

        # Amount of hidden units per layer
        self.n_hidden = n_hidden

        # Amount of layers
        self.n_layers = len(n_hidden)

        # Number of steps Gibbs' sampling steps
        self.steps = steps

        # Learning rate
        self.lr = learning_rate

        # Momentum parameter
        self.momentum = momentum

        # Weight decay
        self.decay = decay

        # Temperature factor
        self.T = temperature

        # Which Fourier components to use
        self.fr_features = fr_features

        self.models = []

        for i in range(self.n_layers):
            if model[i]=='fourier':
                m = FRRBM(self.n_visible, n_hidden[i], steps[i], learning_rate[i],
			momentum[i], decay[i], temperature[i], use_gpu, fr_features[i])
            elif model[i]=='gaussian':
                m = GaussianRBM(self.n_visible, n_hidden[i], steps[i], learning_rate[i], momentum[i], decay[i], temperature[i], use_gpu)

            else:
                m = SigmoidRBM(self.n_visible, n_hidden[i], steps[i], learning_rate[i], momentum[i], decay[i], temperature[i], use_gpu)

            self.models.append(m)

        # Checks if current device is CUDA-based
        if self.device == 'cuda':
            # If yes, uses CUDA in the whole class
            self.cuda()

        logger.info('Class overrided.')


    def fit(self, dataset, batch_size=128, epochs=10):
        """Fits a new MultFRBM model.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the training data.
            batch_size (int): Amount of samples per batch.
            epochs (int): Number of training epochs.

        Returns:
            MSE (mean squared error) and log pseudo-likelihood from the training step.

        """

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=0)

        # For every epoch
        for epoch in range(epochs):
            logger.info('Epoch %d/%d', epoch+1, epochs)

            # Calculating the time of the epoch's starting
            start = time.time()

            # For every epoch
            for j in range(self.n_layers):
                self.models[j].fit(dataset, batch_size, 1)

            # Calculating the time of the epoch's ending
            end = time.time()
            mse = self.models[0].history['mse'][epoch]
            pl = self.models[0].history['pl'][epoch]
            for j in range(1, self.n_layers):
                mse += self.models[j].history['mse'][epoch]
                pl += self.models[j].history['pl'][epoch]

            # Dumps the desired variables to the model's history
            self.dump(mse=mse, pl=pl, time=end-start)

            logger.info('MSE: %f | log-PL: %f', mse, pl)

        return mse, pl

    def reconstruct(self, dataset):
        """Reconstructs batches of new samples.

        Args:
            dataset (torch.utils.data.Dataset): A Dataset object containing the testing data.

        Returns:
            Reconstruction error and visible probabilities, i.e., P(v|h).

        """

        logger.info('Reconstructing new samples ...')
        
	# NEEDS TO BE REFURMULATED! #

        # Resetting MSE to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

        mse, ftran, orig = self.models[0].reconstruct(dataset)
        mse2, samples = self.models[1].reconstruct(dataset)

        logger.info('MSE: %f %f', mse.item(), mse2.item())

        return [mse, mse2], [ftran, samples], dataset.data

    def forward(self, x):
        """Performs a forward pass over the data.

        Args:
            x (torch.Tensor): An input tensor for computing the forward pass.

        Returns:
            A tensor containing the RBM's outputs.

        """

        ftran = self.models[0].forward(x).detach()
        samples = []
        for i in range(1, self.n_layers):
            samples.append(self.models[i].forward(x).detach())

        for j in range(1, self.n_layers):
            ftran = torch.cat((ftran, samples[j-1]), -1)

        return ftran
