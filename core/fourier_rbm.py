"""Fourier-based Gaussian-Bernoulli Restricted Boltzmann Machine.
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
from learnergy.models.gaussian.gaussian_rbm import GaussianRBM

from torch.fft import fftn, ifftn
#from learnergy.utils.fft_utils import fftshift, ifftshift

logger = l.get_logger(__name__)


class FRRBM(GaussianRBM):
    """A FourierRBM class provides the basic implementation for
    Gaussian-Bernoulli Restricted Boltzmann Machines (with standardization).

    This is a trick to ease the calculations of the hidden and
    visible layer samplings, as well as the cost function.

    References:
        Roder M, de Rosa G. H.
	...(2021)

    """

    def __init__(self, n_visible=128, n_hidden=128, steps=1, learning_rate=0.001,
                 momentum=0, decay=0, temperature=1, use_gpu=False, fr_features='phase'):
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

        self.fr_features = fr_features

        logger.info('Overriding class: GaussianRBM -> FRRBM'+'_'+fr_features)

        # Override its parent class
        super(FRRBM, self).__init__(n_visible, n_hidden, steps, learning_rate,
                                          momentum, decay, temperature, use_gpu)

        logger.info('Class overrided.')


    def fit(self, dataset, batch_size=128, epochs=10):
        """Fits a new RBM model.

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

            # Resetting epoch's MSE and pseudo-likelihood to zero
            mse = 0
            pl = 0

            # For every batch
            for samples, _ in tqdm(batches):
                ftran = fftn(samples)#.squeeze()
                #ftran = fftshift(ftran)[:,:,:,0]

                if self.fr_features == 'phase':
                    ftran = torch.angle(ftran.squeeze())

                if self.fr_features == 'magnitude':
                    ftran = torch.abs(ftran.squeeze())

                # Flattening the samples' batch
                samples = ftran.reshape(len(ftran), self.n_visible)
                samples = ((samples - torch.mean(samples, 0, True)) / (torch.std(samples, 0, True) + c.EPSILON)).detach()

                # Checking whether GPU is avaliable and if it should be used
                if self.device == 'cuda':
                    # Applies the GPU usage to the data
                    samples = samples.cuda()

                # Performs the Gibbs sampling procedure
                _, _, _, _, visible_states = self.gibbs_sampling(samples)

                # Detaching the visible states from GPU for further computation
                visible_states = visible_states.detach()

                # Calculates the loss for further gradients' computation
                cost = torch.mean(self.energy(samples)) - \
                    torch.mean(self.energy(visible_states))

                # Initializing the gradient
                self.optimizer.zero_grad()

                # Computing the gradients
                cost.backward()

                # Updating the parameters
                self.optimizer.step()

                # Gathering the size of the batch
                batch_size = samples.size(0)

                # Calculating current's batch MSE
                batch_mse = torch.div(
                    torch.sum(torch.pow(samples - visible_states, 2)), batch_size).detach()

                # Calculating the current's batch logarithm pseudo-likelihood
                batch_pl = self.pseudo_likelihood(samples).detach()

                # Summing up to epochs' MSE and pseudo-likelihood
                mse += batch_mse
                pl += batch_pl

            # Normalizing the MSE and pseudo-likelihood with the number of batches
            mse /= len(batches)
            pl /= len(batches)

            # Calculating the time of the epoch's ending
            end = time.time()

            # Dumps the desired variables to the model's history
            self.dump(mse=mse.item(), pl=pl.item(), time=end-start)

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

        # Resetting MSE to zero
        mse = 0

        # Defining the batch size as the amount of samples in the dataset
        batch_size = len(dataset)

        # Transforming the dataset into training batches
        batches = DataLoader(dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

        # For every batch
        for samples, _ in tqdm(batches):
            ftran = fftn(samples)#.squeeze()

            if self.fr_features == 'phase':
                ftran = torch.angle(ftran.squeeze())

            if self.fr_features == 'magnitude':
                ftran = torch.abs(ftran.squeeze())

            # Flattening the samples' batch
            samples = ftran.reshape(len(ftran), self.n_visible)
            samples = ((samples - torch.mean(samples, 0, True)) / (torch.std(samples, 0, True) + c.EPSILON)).detach()

            # Checking whether GPU is avaliable and if it should be used
            if self.device == 'cuda':
                # Applies the GPU usage to the data
                samples = samples.cuda()

            # Calculating positive phase hidden probabilities and states
            _, pos_hidden_states = self.hidden_sampling(samples)

            # Calculating visible probabilities and states
            visible_probs, visible_states = self.visible_sampling(
                pos_hidden_states)

            # Calculating current's batch reconstruction MSE
            batch_mse = torch.div(
                torch.sum(torch.pow(samples - visible_states, 2)), batch_size)

            # Summing up the reconstruction's MSE
            mse += batch_mse

        # Normalizing the MSE with the number of batches
        mse /= len(batches)

        logger.info('MSE: %f', mse)

        return mse, visible_probs, samples

    def forward(self, x):
        """Performs a forward pass over the data.

        Args:
            x (torch.Tensor): An input tensor for computing the forward pass.

        Returns:
            A tensor containing the RBM's outputs.

        """

        ftran = fftn(x)

        if self.fr_features == 'phase':
            ftran = torch.angle(ftran.squeeze())

        if self.fr_features == 'magnitude':
            ftran = torch.abs(ftran.squeeze())

        # Flattening the samples' batch
        x = ftran.reshape(len(ftran), self.n_visible)
        x = ((x - torch.mean(x, 0, True)) / (torch.std(x, 0, True) + c.EPSILON)).detach()

        # Calculates the outputs of the model
        x, _ = self.hidden_sampling(x)

        return x.detach()


