# Copyright 2023 Álvaro Goldar Dieste

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Neural networks for remote sensing classification.

This module contains various neural architectures for remote sensing classification:

- A simple CNN 2D.
- A residual CNN 2D.
- An ACGAN built on top of the simple CNN 2D.
- An ACGAN built on top of the residual CNN 2D.
- A BAGAN built on top of the simple CNN 2D.
- A BAGAN built on top of the residual CNN 2D.

This module also provides all neccessary functions to train, validate and test any of these architectures.
"""


__author__ = "alvrogd"


import math
import sys

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.profiler
import torch.nn as nn
import torch.nn.functional as functional
import torchsummary
import torchvision.utils
import tqdm


class CNN2D(nn.Module):
    """Simple 2D Convolutional Neuronal Network.

    This network is composed of 4 convolutional blocks and 2 fully-connected blocks.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    Z : int
        The size of the latent space.

    classifier_weights : torch.Tensor
        The weights that will be used to compute the loss function.

    classifier : torch.nn._WeightedLoss
        The loss function that will be used to train the network.

    optimizer : torch.optim.Opimizer
        The optimizer that will be used to train the network.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(CNN2D, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device
        self.Z             = hyperparams["latent_size"]


        # 1. The first convolutional block changes the number of channels to 16
        self.conv_1 = GAN_Discriminator_ConvBlock(hyperparams, self.bands, 16, norm=False)

        # 2. The second convolutional block changes the number of channels to 32
        #    It also downsamples the spatial resolution by a factor of 2
        self.conv_2 = GAN_Discriminator_ConvBlock(hyperparams, 16, 32, strides=(2, 2), norm=False)

        # 3. The third convolutional block changes the number of channels to 64
        #    It also downsamples the spatial resolution by a factor of 2
        self.conv_3 = GAN_Discriminator_ConvBlock(hyperparams, 32, 64, strides=(2, 2), norm=False)

        # 4. The fourth convolutional block changes the number of channels to Z
        #    That is the same number of channels as the latent space size used on the generator
        #    It also downsamples the spatial resolution by a factor of 2
        self.conv_4 = GAN_Discriminator_ConvBlock(hyperparams, 64, self.Z, strides=(2, 2), norm=False)

        # 5. Now we apply a fully-connected layer to the flattened output of the last convolutional block to get the
        #    final feature vector
        #    Each sample has been downsampled to 4x4 pixels
        self.full_1 = GAN_Discriminator_LinearBlock(hyperparams, 4 * 4 * self.Z, self.Z)

        # 6. The final layer is another fully connected layer that outputs the class probabilities
        self.full_2 = nn.Linear(self.Z, self.classes_count)


        self.apply(self.initialize_weights)


        # From PyTorch docs:
        #   "If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
        #    Parameters of a model after .cuda() will be different objects with those before the call."
        self.to(self.device)


        # Each class is given the same weight
        self.classifier_weights = torch.ones(self.classes_count, device=self.device)
        self.classifier         = nn.CrossEntropyLoss(weight=self.classifier_weights)


        self.optimizer = torch.optim.Adam(self.parameters(), lr=hyperparams["learning_rate"], betas=(0.5, 0.999))


    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        tuple
            The result of applying the network's different transformations over the input batch.

            More specifically, the output is a tuple containing the following tensors:

            - The output of the last layer (class probabilities).
            - The features that are used to compute the output.
        """

        out = self.conv_1(tensor)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)

        features = out.view(out.shape[0], -1)
        features = self.full_1(features)
        logits   = self.full_2(features)
        # The softmax function is applied outside

        return (logits, features)


    @staticmethod
    def initialize_weights(child):
        """Receives a layer and initializes its weights and biases if needed.

        Parameters
        ----------
        child : torch.nn.Module
            The layer.
        """

        if isinstance(child, nn.Conv2d):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)

        elif isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)


    def __str__(self):
        return "[*] CNN2D network summary:\n" \
               f"{torchsummary.summary(self, [self.bands, self.patch_size, self.patch_size], verbose=0, device=self.device)}"


class CNN2D_Residual(nn.Module):
    """Residual 2D Convolutional Neuronal Network.

    This network is composed of 3 residual stages followed by an average pooling layer and a fully-connected block.
    Each residual stage is composed of 3 convolutional blocks. All convolutional blocks in a given stage use the same
    number of channels; this amount is increased with each stage. A feature-fusion that combines the features
    extracted from each stage is performed before the average pooling layer.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    Z : int
        The size of the latent space.

    classifier_weights : torch.Tensor
        The weights that will be used to compute the loss function.

    classifier : torch.nn._WeightedLoss
        The loss function that will be used to train the network.

    optimizer : torch.optim.Opimizer
        The optimizer that will be used to train the network.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(CNN2D_Residual, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device
        self.Z             = hyperparams["latent_size"]


        # 0. A mapping block changes the number of channels to 16
        self.map_0 = GAN_Discriminator_ConvBlock(hyperparams, self.bands, 16, old_norm=True)
        
        # 1. The first residual stage consists of three residual blocks that use 16 channels
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_1 = GAN_Discriminator_ResStage(hyperparams, 3, 16, 16, strides=(2, 2), old_norm=True)

        # 2. The second residual stage consists of three residual blocks that use 32 channels
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_2 = GAN_Discriminator_ResStage(hyperparams, 3, 16, 32, strides=(2, 2), old_norm=True)

        # 3. The third residual stage consists of three residual blocks that use Z channels
        #    That is the same number of channels as the latent space size used on the generator
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_3 = GAN_Discriminator_ResStage(hyperparams, 3, 32, self.Z, strides=(2, 2), old_norm=True)

        # 4. In order to fuse features from different stages, we need two additional convolutional blocks that map the
        #    low-resolution features to the high-resolution ones
        self.map_stage1 = GAN_Discriminator_ConvBlock(hyperparams, 16, self.Z, strides=(4, 4), old_norm=True)
        self.map_stage2 = GAN_Discriminator_ConvBlock(hyperparams, 32, self.Z, strides=(2, 2), old_norm=True)

        # 5. Now we apply an average pooling layer to get the final feature vector
        #    Each sample has been downsampled to 4x4 pixels
        self.avg_pool = nn.AvgPool2d((4, 4))

        # 6. The final layer is a fully connected layer that outputs the class probabilities
        self.full = nn.Linear(self.Z, self.classes_count)


        self.apply(self.initialize_weights)


        # From PyTorch docs:
        #   "If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
        #    Parameters of a model after .cuda() will be different objects with those before the call."
        self.to(self.device)


        # Each class is given the same weight
        self.classifier_weights = torch.ones(self.classes_count, device=self.device)
        self.classifier         = nn.CrossEntropyLoss(weight=self.classifier_weights)


        self.optimizer = torch.optim.Adam(self.parameters(), lr=hyperparams["learning_rate"], betas=(0.5, 0.999))


    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        tuple
            The result of applying the network's different transformations over the input batch.

            More specifically, the output is a tuple containing the following tensors:

            - The output of the last layer (class probabilities).
            - The features that are used to compute the output.
        """

        out_start  = self.map_0(tensor)
        
        out_stage1 = self.stage_1(out_start)
        out_stage2 = self.stage_2(out_stage1)
        out_stage3 = self.stage_3(out_stage2)

        out = self.map_stage1(out_stage1) + self.map_stage2(out_stage2) + out_stage3
        out = self.avg_pool(out)

        features = out.view(out.shape[0], -1)
        logits   = self.full(features)
        # The softmax function is applied outside

        return (logits, features)


    @staticmethod
    def initialize_weights(child):
        """Receives a layer and initializes its weights and biases if needed.

        Parameters
        ----------
        child : torch.nn.Module
            The layer.
        """

        if isinstance(child, nn.Conv2d):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)

        elif isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)


    def __str__(self):
        return "[*] CNN2D_Residual network summary:\n" \
               f"{torchsummary.summary(self, [self.bands, self.patch_size, self.patch_size], verbose=0, device=self.device)}"


class ACGAN(nn.Module):
    """Auxiliary Classifier Generative Adversarial Networks.

    This network is composed of a generator and a discriminator. The discriminator is the Simple 2D Convolutional
    Neuronal Network, while the generator is another simple CNN.

    During training, the generator is trained to fool the discriminator, while the discriminator is trained to
    distinguish between real and fake samples and to classify them.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    discriminator : torch.nn.Module
        The discriminator network.

    generator : torch.nn.Module
        The generator network.

    loss : torch.nn._WeightedLoss
        The loss function used to train the network with regards to real and fake sample detection.

    classifier_weights : torch.Tensor
        The weights that will be used to compute the classification loss function.

    classifier : torch.nn._WeightedLoss
        The loss function that will be used to train the network with regards to the classification task.

    optimizer_D : torch.optim.Opimizer
        The optimizer that will be used to train the discriminator.

    optimizer_G : torch.optim.Opimizer
        The optimizer that will be used to train the generator.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(ACGAN, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device


        # Each network can be executed independently
        self.discriminator = ACGAN_Discriminator(dataset, self.device, hyperparams)
        self.generator     = GAN_Generator(dataset, self.device, hyperparams)


        # Each network takes care of initializing their own weights


        # From PyTorch docs:
        #   "If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
        #    Parameters of a model after .cuda() will be different objects with those before the call."
        self.to(self.device)


        # Binary Cross Entropy loss is used to check whether the discriminator correctly tells fake samples apart from
        # real ones
        self.loss = nn.BCELoss()

        # Each class is given the same weight
        # --> this object is shared between the discriminator and the generator
        self.classifier_weights = torch.ones(self.classes_count, device=self.device)
        self.classifier         = nn.CrossEntropyLoss(weight=self.classifier_weights)


        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=hyperparams["learning_rate"], \
                                            betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=hyperparams["learning_rate"], \
                                            betas=(0.5, 0.999))


    def __str__(self):
        return "[*] ACGAN network summary:\n" \
               f"{self.discriminator}\n" \
               f"{self.generator}\n"
               
               
class ResACGAN(nn.Module):
    """Residual Auxiliary Classifier Generative Adversarial Networks.

    This network is composed of a generator and a discriminator. The discriminator is the Residual 2D Convolutional
    Neuronal Network, while the generator is a simple CNN.

    During training, the generator is trained to fool the discriminator, while the discriminator is trained to
    distinguish between real and fake samples and to classify them.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    discriminator : torch.nn.Module
        The discriminator network.

    generator : torch.nn.Module
        The generator network.

    loss : torch.nn._WeightedLoss
        The loss function used to train the network with regards to real and fake sample detection.

    classifier_weights : torch.Tensor
        The weights that will be used to compute the classification loss function.

    classifier : torch.nn._WeightedLoss
        The loss function that will be used to train the network with regards to the classification task.

    optimizer_D : torch.optim.Opimizer
        The optimizer that will be used to train the discriminator.

    optimizer_G : torch.optim.Opimizer
        The optimizer that will be used to train the generator.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(ResACGAN, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device


        # Each network can be executed independently
        self.discriminator = ResACGAN_Discriminator(dataset, self.device, hyperparams)
        self.generator     = GAN_Generator(dataset, self.device, hyperparams)


        # Each network takes care of initializing their own weights


        # From PyTorch docs:
        #   "If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
        #    Parameters of a model after .cuda() will be different objects with those before the call."
        self.to(self.device)


        # Binary Cross Entropy loss is used to check whether the discriminator correctly tells fake samples apart from
        # real ones
        self.loss = nn.BCELoss()

        # Each class is given the same weight
        # --> this object is shared between the discriminator and the generator
        self.classifier_weights = torch.ones(self.classes_count, device=self.device)
        self.classifier         = nn.CrossEntropyLoss(weight=self.classifier_weights)


        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=hyperparams["learning_rate"], \
                                            betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=hyperparams["learning_rate"], \
                                            betas=(0.5, 0.999))


    def __str__(self):
        return "[*] ResACGAN network summary:\n" \
               f"{self.discriminator}\n" \
               f"{self.generator}\n"


class BAGAN(nn.Module):
    """
    Balancing Generative Adversarial Networks.

    This network is composed of two sub-networks: an autoencoder and a GAN. To train this network, the autoencoder is
    used first to generate the initial understanding of data distribution. Then, its parameters are used to initialize
    the GAN. At last, the GAN is trained as any other GAN.

    The GAN sub-network is composed of a generator and a discriminator, which share the same structure as the decoder
    and encoder of the autoencoder, respectively. The discriminator is the Simple 2D Convolutional Neuronal Network,
    while the generator is another simple CNN.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    autoencoder : torch.nn.Module
        The autoencoder network.

    discriminator : torch.nn.Module
        The discriminator network.

    generator : torch.nn.Module
        The generator network.

    criterion_A : torch.nn._Loss
        The loss function that will be used to train the autoencoder.

    classifier_weights : torch.Tensor
        The weights that will be used to compute the loss function of the GAN.

    classifier : torch.nn._WeightedLoss
        The loss function that will be used to train the GAN.

    optimizer_A : torch.optim.Opimizer
        The optimizer that will be used to train the autoencoder.

    optimizer_D : torch.optim.Opimizer
        The optimizer that will be used to train the discriminator.

    optimizer_G : torch.optim.Opimizer
        The optimizer that will be used to train the generator.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(BAGAN, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device


        # Each network can be executed independently
        self.autoencoder   = BAGAN_Autoencoder(dataset, self.device, hyperparams)
        self.discriminator = BAGAN_Discriminator(dataset, self.device, hyperparams)
        self.generator     = GAN_Generator(dataset, self.device, hyperparams)


        # Each network takes care of initializing their own weights


        # From PyTorch docs:
        #   "If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
        #    Parameters of a model after .cuda() will be different objects with those before the call."
        self.to(self.device)


        # Mean squared error loss is used to evaluate the similarity between the real images and the reconstructed
        # ones
        self.criterion_A = nn.MSELoss()

        # Each class is given the same weight
        # --> keep in mind that there is an additional "fake" class
        # --> this object is shared between the discriminator and the generator
        self.classifier_weights = torch.ones(self.classes_count + 1, device=self.device)
        self.classifier         = nn.CrossEntropyLoss(weight=self.classifier_weights)


        self.optimizer_A = torch.optim.Adam(self.autoencoder.parameters(), lr=hyperparams["learning_rate"])
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=hyperparams["learning_rate"], \
                                            betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=hyperparams["learning_rate"], \
                                            betas=(0.5, 0.999))


    def __str__(self):
        return "[*] BAGAN network summary:\n" \
               f"{self.autoencoder}\n" \
               f"{self.discriminator}\n" \
               f"{self.generator}\n"


class ResBaGAN(nn.Module):
    """
    Residual Balancing Generative Adversarial Networks.

    This network is composed of two sub-networks: an autoencoder and a GAN. To train this network, the autoencoder is
    used first to generate the initial understanding of data distribution. Then, its parameters are used to initialize
    the GAN. At last, the GAN is trained as any other GAN.

    The GAN sub-network is composed of a generator and a discriminator, which share the same structure as the decoder
    and encoder of the autoencoder, respectively. The discriminator is the Residual 2D Convolutional Neuronal Network,
    while the generator is a simple CNN.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    autoencoder : torch.nn.Module
        The autoencoder network.

    discriminator : torch.nn.Module
        The discriminator network.

    generator : torch.nn.Module
        The generator network.

    criterion_A : torch.nn._Loss
        The loss function that will be used to train the autoencoder.

    classifier_weights : torch.Tensor
        The weights that will be used to compute the loss function of the GAN.

    classifier : torch.nn._WeightedLoss
        The loss function that will be used to train the GAN.

    optimizer_A : torch.optim.Opimizer
        The optimizer that will be used to train the autoencoder.

    optimizer_D : torch.optim.Opimizer
        The optimizer that will be used to train the discriminator.

    optimizer_G : torch.optim.Opimizer
        The optimizer that will be used to train the generator.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(ResBaGAN, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device


        # Each network can be executed independently
        self.autoencoder   = ResBaGAN_Autoencoder(dataset, self.device, hyperparams)
        self.discriminator = ResBaGAN_Discriminator(dataset, self.device, hyperparams)
        self.generator     = GAN_Generator(dataset, self.device, hyperparams)


        # Each network takes care of initializing their own weights


        # From PyTorch docs:
        #   "If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
        #    Parameters of a model after .cuda() will be different objects with those before the call."
        self.to(self.device)


        # Mean squared error loss is used to evaluate the similarity between the real images and the reconstructed
        # ones
        self.criterion_A = nn.MSELoss()

        # Each class is given the same weight
        # --> keep in mind that there is an additional "fake" class
        # --> this object is shared between the discriminator and the generator
        self.classifier_weights = torch.ones(self.classes_count + 1, device=self.device)
        self.classifier         = nn.CrossEntropyLoss(weight=self.classifier_weights)


        self.optimizer_A = torch.optim.Adam(self.autoencoder.parameters(), lr=hyperparams["learning_rate"])
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=hyperparams["learning_rate"], \
                                            betas=(0.5, 0.999))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=hyperparams["learning_rate"], \
                                            betas=(0.5, 0.999))


    def __str__(self):
        return "[*] ResBaGAN network summary:\n" \
               f"{self.autoencoder}\n" \
               f"{self.discriminator}\n" \
               f"{self.generator}\n"


class BAGAN_Autoencoder(nn.Module):
    """Autoencoder for BAGAN.

    This network is composed of an encoder and a decoder. The encoder is the Simple 2D Convolutional Neuronal Network,
    while the decoder is another simple CNN.

    During training, the encoder summarizes the input data, while the decoder is trained to reconstruct the data from
    the summaries as accurately as possible.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    discriminator : torch.nn.Module
        The encoder network.

    generator : torch.nn.Module
        The decoder network.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(BAGAN_Autoencoder, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device


        self.discriminator = BAGAN_Discriminator(dataset, self.device, hyperparams)
        self.generator     = GAN_Generator(dataset, self.device, hyperparams)


        # Each network takes care of initializing their own weights


    def forward(self, tensors):
        """Processes a batch of samples.

        Parameters
        ----------
        tensors : tuple
            A tuple containing (1) the input samples and (2) the corresponding label for each one.

        Returns
        -------
        torch.Tensor
            The result of applying the network's different transformations over the input batch.

            More specifically, the reconstructions of the input samples.
        """

        if isinstance(tensors, tuple):
            image, label = tensors
        else:
            # nasty fix to be able to run torchsummary and tensorboard
            image, label = tensors, torch.zeros((2), dtype=torch.int64, device=self.device)

        _, features    = self.discriminator(image)
        reconstruction = self.generator((features, label))

        return reconstruction


    def __str__(self):
        return "[*] BAGAN_Autoencoder network summary:\n" \
               f"Contains both a BAGAN_Discriminator and GAN_Generator"


class ResBaGAN_Autoencoder(nn.Module):
    """Autoencoder for ResBaGAN.

    This network is composed of an encoder and a decoder. The encoder is the Residual 2D Convolutional Neuronal
    Network, while the decoder is a simple CNN.

    During training, the encoder summarizes the input data, while the decoder is trained to reconstruct the data from
    the summaries as accurately as possible.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    discriminator : torch.nn.Module
        The encoder network.

    generator : torch.nn.Module
        The decoder network.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(ResBaGAN_Autoencoder, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device


        self.discriminator = ResBaGAN_Discriminator(dataset, self.device, hyperparams)
        self.generator     = GAN_Generator(dataset, self.device, hyperparams)


        # Each network takes care of initializing their own weights


    def forward(self, tensors):
        """Processes a batch of samples.

        Parameters
        ----------
        tensors : tuple
            A tuple containing (1) the input samples and (2) the corresponding label for each one.

        Returns
        -------
        torch.Tensor
            The result of applying the network's different transformations over the input batch.

            More specifically, the reconstructions of the input samples.
        """

        if isinstance(tensors, tuple):
            image, label = tensors
        else:
            # nasty fix to be able to run torchsummary and tensorboard
            image, label = tensors, torch.zeros((2), dtype=torch.int64, device=self.device)

        _, features    = self.discriminator(image)
        reconstruction = self.generator((features, label))

        return reconstruction


    def __str__(self):
        return "[*] ResBaGAN_Autoencoder network summary:\n" \
               f"Contains both a ResBaGAN_Discriminator and GAN_Generator"


class ACGAN_Discriminator(nn.Module):
    """Discriminator for ACGAN.

    This network is equivalent to the Simple 2D Convolutional Neuronal Network. The difference is that it has two
    outputs: one for the classification task and another for the fake samples detection task.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    Z : int
        The size of the latent space.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(ACGAN_Discriminator, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device
        self.Z             = hyperparams["latent_size"]


        # 1. The first convolutional block changes the number of channels to 16
        self.conv_1 = GAN_Discriminator_ConvBlock(hyperparams, self.bands, 16)

        # 2. The second convolutional block changes the number of channels to 32
        #    It also downsamples the spatial resolution by a factor of 2
        self.conv_2 = GAN_Discriminator_ConvBlock(hyperparams, 16, 32, strides=(2, 2))

        # 3. The third convolutional block changes the number of channels to 64
        #    It also downsamples the spatial resolution by a factor of 2
        self.conv_3 = GAN_Discriminator_ConvBlock(hyperparams, 32, 64, strides=(2, 2))

        # 4. The fourth convolutional block changes the number of channels to Z
        #    That is the same number of channels as the latent space size used on the generator
        #    It also downsamples the spatial resolution by a factor of 2
        self.conv_4 = GAN_Discriminator_ConvBlock(hyperparams, 64, self.Z, strides=(2, 2))

        # 5. Now we apply a fully-connected layer to the flattened output of the last convolutional block to get the
        #    final feature vector
        #    Each sample has been downsampled to 4x4 pixels
        self.full_1 = GAN_Discriminator_LinearBlock(hyperparams, 4 * 4 * self.Z, self.Z)

        # 6. The final layer is another fully connected layer that outputs the class probabilities
        self.full_2 = nn.Linear(self.Z, self.classes_count)

        #    An additional final layer to tell fake and real samples apart
        self.full_lies = nn.Sequential(
            nn.Linear(self.Z, 1),
            nn.Sigmoid()
        )


        self.apply(self.initialize_weights)


    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        tuple
            The result of applying the network's different transformations over the input batch.

            More specifically, the output is a tuple containing the following tensors:

            - The class probabilities.
            - The features that are used to compute the outputs.
            - The probability of each sample being real.
        """

        out = self.conv_1(tensor)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)

        features = out.view(out.shape[0], -1)
        features = self.full_1(features)
        logits   = self.full_2(features)
        # The softmax function is applied outside on logits
        lies     = self.full_lies(features)


        return (logits, features, lies)


    @staticmethod
    def initialize_weights(child):
        """Receives a layer and initializes its weights and biases if needed.

        Parameters
        ----------
        child : torch.nn.Module
            The layer.
        """

        if isinstance(child, nn.Conv2d):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)

        elif isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)


    def __str__(self):
        return "[*] ACGAN_Discriminator network summary:\n" \
               f"{torchsummary.summary(self, [self.bands, self.patch_size, self.patch_size], verbose=0, device=self.device)}"


class ResACGAN_Discriminator(nn.Module):
    """Discriminator for ResACGAN.

    This network is equivalent to the Residual 2D Convolutional Neuronal Network. The difference is that it has two
    outputs: one for the classification task and another for the fake samples detection task.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    Z : int
        The size of the latent space.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(ResACGAN_Discriminator, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device
        self.Z             = hyperparams["latent_size"]


        # 0. A mapping block changes the number of channels to 16
        self.map_0 = GAN_Discriminator_ConvBlock(hyperparams, self.bands, 16)
        
        # 1. The first residual stage consists of three residual blocks that use 16 channels
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_1 = GAN_Discriminator_ResStage(hyperparams, 3, 16, 16, strides=(2, 2))

        # 2. The second residual stage consists of three residual blocks that use 32 channels
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_2 = GAN_Discriminator_ResStage(hyperparams, 3, 16, 32, strides=(2, 2))

        # 3. The third residual stage consists of three residual blocks that use Z channels
        #    That is the same number of channels as the latent space size used on the generator
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_3 = GAN_Discriminator_ResStage(hyperparams, 3, 32, self.Z, strides=(2, 2))

        # 4. In order to fuse features from different stages, we need two additional convolutional blocks that map the
        #    low-resolution features to the high-resolution ones
        self.map_stage1 = GAN_Discriminator_ConvBlock(hyperparams, 16, self.Z, strides=(4, 4))
        self.map_stage2 = GAN_Discriminator_ConvBlock(hyperparams, 32, self.Z, strides=(2, 2))

        # 5. Now we apply an average pooling layer to get the final feature vector
        #    Each sample has been downsampled to 4x4 pixels
        self.avg_pool = nn.AvgPool2d((4, 4))

        # 6. The final layer is another fully connected layer that outputs the class probabilities
        self.full_2 = nn.Linear(self.Z, self.classes_count)

        #    An additional final layer to tell fake and real samples apart
        self.full_lies = nn.Sequential(
            nn.Linear(self.Z, 1),
            nn.Sigmoid()
        )


        self.apply(self.initialize_weights)


    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        tuple
            The result of applying the network's different transformations over the input batch.

            More specifically, the output is a tuple containing the following tensors:

            - The class probabilities.
            - The features that are used to compute the outputs.
            - The probability of each sample being real.
        """
        
        out_start  = self.map_0(tensor)
        
        out_stage1 = self.stage_1(out_start)
        out_stage2 = self.stage_2(out_stage1)
        out_stage3 = self.stage_3(out_stage2)

        out = self.map_stage1(out_stage1) + self.map_stage2(out_stage2) + out_stage3
        out = self.avg_pool(out)

        features = out.view(out.shape[0], -1)
        logits   = self.full_2(features)
        # The softmax function is applied outside on logits
        lies     = self.full_lies(features)

        return (logits, features, lies)


    @staticmethod
    def initialize_weights(child):
        """Receives a layer and initializes its weights and biases if needed.

        Parameters
        ----------
        child : torch.nn.Module
            The layer.
        """

        if isinstance(child, nn.Conv2d):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)

        elif isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)


    def __str__(self):
        return "[*] ResACGAN_Discriminator network summary:\n" \
               f"{torchsummary.summary(self, [self.bands, self.patch_size, self.patch_size], verbose=0, device=self.device)}"


class BAGAN_Discriminator(nn.Module):
    """Discriminator for BAGAN.

    This network is equivalent to the Simple 2D Convolutional Neuronal Network, but its output includes an additional
    class to identify fake samples.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    Z : int
        The size of the latent space.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(BAGAN_Discriminator, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device
        self.Z             = hyperparams["latent_size"]


        # 1. The first convolutional block changes the number of channels to 16
        self.conv_1 = GAN_Discriminator_ConvBlock(hyperparams, self.bands, 16)

        # 2. The second convolutional block changes the number of channels to 32
        #    It also downsamples the spatial resolution by a factor of 2
        self.conv_2 = GAN_Discriminator_ConvBlock(hyperparams, 16, 32, strides=(2, 2))

        # 3. The third convolutional block changes the number of channels to 64
        #    It also downsamples the spatial resolution by a factor of 2
        self.conv_3 = GAN_Discriminator_ConvBlock(hyperparams, 32, 64, strides=(2, 2))

        # 4. The fourth convolutional block changes the number of channels to Z
        #    That is the same number of channels as the latent space size used on the generator
        #    It also downsamples the spatial resolution by a factor of 2
        self.conv_4 = GAN_Discriminator_ConvBlock(hyperparams, 64, self.Z, strides=(2, 2))

        # 5. Now we apply a fully-connected layer to the flattened output of the last convolutional block to get the
        #    final feature vector
        #    Each sample has been downsampled to 4x4 pixels
        self.full_1 = GAN_Discriminator_LinearBlock(hyperparams, 4 * 4 * self.Z, self.Z)

        # 6. The final layer is another fully connected layer that outputs the class probabilities
        self.full_2 = nn.Linear(self.Z, self.classes_count + 1)


        self.apply(self.initialize_weights)


    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        tuple
            The result of applying the network's different transformations over the input batch.

            More specifically, the output is a tuple containing the following tensors:

            - The class probabilities.
            - The features that are used to compute the output.
        """

        out = self.conv_1(tensor)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = self.conv_4(out)

        features = out.view(out.shape[0], -1)
        features = self.full_1(features)
        logits   = self.full_2(features)
        # The softmax function is applied outside

        return (logits, features)


    @staticmethod
    def initialize_weights(child):
        """Receives a layer and initializes its weights and biases if needed.

        Parameters
        ----------
        child : torch.nn.Module
            The layer.
        """

        if isinstance(child, nn.Conv2d):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)

        elif isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)


    def __str__(self):
        return "[*] BAGAN_Discriminator network summary:\n" \
               f"{torchsummary.summary(self, [self.bands, self.patch_size, self.patch_size], verbose=0, device=self.device)}"


class ResBaGAN_Discriminator(nn.Module):
    """Discriminator for ResBaGAN.

    This network is equivalent to the Residual 2D Convolutional Neuronal Network, but its output includes an
    additional class to identify fake samples.

    Attributes
    ----------
    bands : int
        The number of bands in the input samples.

    patch_size : int
        The size of the input samples. It is measured in pixels along the patch's side.

    classes_count : int
        The number of classes in the target dataset.

    device : str
        The computational device that will be used to run the network.

    Z : int
        The size of the latent space.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(ResBaGAN_Discriminator, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device
        self.Z             = hyperparams["latent_size"]


        # 0. A mapping block changes the number of channels to 16
        self.map_0 = GAN_Discriminator_ConvBlock(hyperparams, self.bands, 16)
        
        # 1. The first residual stage consists of three residual blocks that use 16 channels
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_1 = GAN_Discriminator_ResStage(hyperparams, 3, 16, 16, strides=(2, 2))

        # 2. The second residual stage consists of three residual blocks that use 32 channels
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_2 = GAN_Discriminator_ResStage(hyperparams, 3, 16, 32, strides=(2, 2))

        # 3. The third residual stage consists of three residual blocks that use Z channels
        #    That is the same number of channels as the latent space size used on the generator
        #    It also downsamples the spatial resolution by a factor of 2
        self.stage_3 = GAN_Discriminator_ResStage(hyperparams, 3, 32, self.Z, strides=(2, 2))

        # 4. In order to fuse features from different stages, we need two additional convolutional blocks that map the
        #    low-resolution features to the high-resolution ones
        self.map_stage1 = GAN_Discriminator_ConvBlock(hyperparams, 16, self.Z, strides=(4, 4))
        self.map_stage2 = GAN_Discriminator_ConvBlock(hyperparams, 32, self.Z, strides=(2, 2))

        # 5. Now we apply an average pooling layer to get the final feature vector
        #    Each sample has been downsampled to 4x4 pixels
        self.avg_pool = nn.AvgPool2d((4, 4))

        # 6. The final layer is a fully connected layer that outputs the class probabilities
        self.full = nn.Linear(self.Z, self.classes_count + 1)


        self.apply(self.initialize_weights)


    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        tuple
            The result of applying the network's different transformations over the input batch.

            More specifically, the output is a tuple containing the following tensors:

            - The class probabilities.
            - The features that are used to compute the output.
        """

        out_start  = self.map_0(tensor)
        
        out_stage1 = self.stage_1(out_start)
        out_stage2 = self.stage_2(out_stage1)
        out_stage3 = self.stage_3(out_stage2)

        out = self.map_stage1(out_stage1) + self.map_stage2(out_stage2) + out_stage3
        out = self.avg_pool(out)

        features = out.view(out.shape[0], -1)
        logits   = self.full(features)
        # The softmax function is applied outside

        return (logits, features)


    @staticmethod
    def initialize_weights(child):
        """Receives a layer and initializes its weights and biases if needed.

        Parameters
        ----------
        child : torch.nn.Module
            The layer.
        """

        if isinstance(child, nn.Conv2d):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)

        elif isinstance(child, nn.Linear):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)


    def __str__(self):
        return "[*] ResBaGAN_Discriminator network summary:\n" \
               f"{torchsummary.summary(self, [self.bands, self.patch_size, self.patch_size], verbose=0, device=self.device)}"


class GAN_Discriminator_ResStage(nn.Module):
    """Residual stage for building the residual CNNs in this module.

    A residual stage consists of num_blocks residual blocks that use the same number of channels:

    - The first block changes the number of channels from in_channels to out_channels.
    - The other blocks leave the number of channels unchanged.    

    The first residual block is also able to downsample the spatial dimensions of the input tensor using strides. 

    Attributes
    ----------
    num_blocks : int
        The number of residual blocks in the stage.

    in_channels : int
        The number of channels in the input samples.

    out_channels : int
        The number of channels in the output samples.

    strides : tuple
        The strides used in the first convolutional block of the stage.

    norm : bool
        Whether to use normalization or not after each convolutional block.

    old_norm : bool
        If True, batch normalization is used. Otherwise, spectral normalization is used.
    """

    def __init__(self, hyperparams, num_blocks, in_channels, out_channels, strides=(1, 1), norm=True, old_norm=False):
        """Defines and initializes the residual stage.

        Parameters
        ----------
        hyperparams : dict
            Some hyperparameters are used to define the stage structure and its behavior.

        num_blocks : int
            The number of residual blocks in the stage.

        in_channels : int
            The number of channels in the input samples.

        out_channels : int
            The number of channels in the output samples.

        strides : tuple, optional
            The strides used in the first convolutional block of the stage. Defaults to (1, 1).

        norm : bool, optional
            Whether to use spectral normalization after the convolutional blocks. Defaults to True.

        old_norm : bool, optional
            If True, batch normalization is used instead of spectral normalization. Defaults to
            False.
        """

        super(GAN_Discriminator_ResStage, self).__init__()


        self.num_blocks   = num_blocks
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.strides      = strides
        self.norm         = norm
        self.old_norm     = old_norm


        blocks = []

        # The first convolutional block may:
        #
        #   - Change the number of channels.
        #   - Downsample the spatial dimensions of the input tensor.
        #
        # If any of these is true, we need to provide a downsampling module to the first block so that it is able to
        # apply the residual connection.
        downsample = None

        if (self.in_channels != self.out_channels) or (self.strides != (1, 1)):
            downsample = GAN_Discriminator_ConvBlock(hyperparams, self.in_channels, self.out_channels,
                                                     strides=self.strides, norm=self.norm, lrelu=False,
                                                     old_norm=self.old_norm)

        blocks.append(GAN_Discriminator_ResBlock(hyperparams, self.in_channels, self.out_channels,
                                                 strides=self.strides, downsample=downsample, norm=self.norm,
                                                 old_norm=self.old_norm))


        # The remaining convolutional blocks are simpler
        for _ in range(1, self.num_blocks):
            blocks.append(GAN_Discriminator_ResBlock(hyperparams, self.out_channels, self.out_channels,
                                                     norm=self.norm, old_norm=self.old_norm))


        self.blocks = nn.Sequential(*blocks)


    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        torch.Tensor
            The result of applying the stage's different transformations over the input batch.
        """

        out = self.blocks(tensor)

        return out


class GAN_Discriminator_ResBlock(nn.Module):
    """Residual block for building the residual stages in this module.
    
    A residual block contains two convolutional blocks:
        
    - The first one changes the number of channels from in_channels to out_channels.
    - The second one changes leaves the number of channels unchanged.

    The first convolutional block is also able to downsample the spatial dimensions of the input tensor using strides.

    Attributes
    ----------
    in_channels : int
        The number of channels in the input samples.
    
    out_channels : int
        The number of channels in the output samples.

    strides : tuple
        The strides used in the first convolutional block.

    downsample : torch.nn.Module
        If needed, the module used to downsample the input tensor to perform the residual connection at the end.

    norm : bool
        Whether to use normalization or not after each convolutional block.

    old_norm : bool
        If True, batch normalization is used. Otherwise, spectral normalization is used.
    """

    def __init__(self, hyperparams, in_channels, out_channels, strides=(1, 1), downsample=None, norm=True,
                 old_norm=False):
        """Defines and initializes the residual block.

        Parameters
        ----------
        hyperparams : dict
            Some hyperparameters are used to define the block structure and its behavior.

        in_channels : int
            The number of channels in the input samples.

        out_channels : int
            The number of channels in the output samples.

        strides : tuple, optional
            The strides used in the first convolutional block. Defaults to (1, 1).

        downsample : nn.Module, optional
            The downsampling layer to be used in the residual block, if any. Defaults to None.

        norm : bool, optional
            Whether to use spectral normalization in the convolutional blocks. Defaults to True.

        old_norm : bool, optional
            If True, batch normalization is used instead of spectral normalization. Defaults to
            False.
        """

        super(GAN_Discriminator_ResBlock, self).__init__()


        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.strides      = strides
        self.downsample   = downsample
        self.norm         = norm
        self.old_norm     = old_norm


        # 1. From in_channels to out_channels
        self.conv_1 = GAN_Discriminator_ConvBlock(hyperparams, self.in_channels, self.out_channels,
                                                  strides=self.strides, norm=self.norm, old_norm=self.old_norm)

        # 2. Leaves the same number of channels
        self.conv_2 = GAN_Discriminator_ConvBlock(hyperparams, self.out_channels, self.out_channels, norm=self.norm,
                                                  old_norm=self.old_norm)

        self.downsample = downsample


    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        torch.Tensor
            The result of applying the block's different transformations over the input batch.
        """

        identity = self.downsample(tensor) if self.downsample is not None else tensor

        out =  self.conv_1(tensor)
        out =  self.conv_2(out)

        return out + identity


class GAN_Discriminator_ConvBlock(nn.Module):
    """Convolutional block for building the residual blocks in this module.

    A convolutional block contains a convolutional layer, a normalization layer (optional), an activation layer
    (optional), and a dropout layer (optional).

    Attributes
    ----------
    in_channels : int
        The number of channels in the input samples.
    
    out_channels : int
        The number of channels in the output samples.

    kernel_size : tuple
        The size of the convolutional kernel.

    strides : tuple
        The strides of the convolutional layer.

    norm : bool
        Whether to use normalization or not.

    old_norm : bool
        If True, batch normalization is used. Otherwise, spectral normalization is used.

    lrelu : bool
        Whether to use an activation layer or not.

    dropout : bool
        Whether to use a dropout layer or not.
    """

    def __init__(self, hyperparams, in_channels, out_channels, kernel_size=(3, 3), strides=(1, 1), norm=True,
                 old_norm=False, lrelu=True, dropout=True):
        """Defines and initializes the convolutional block.

        Parameters
        ----------
        hyperparams : dict
            Some hyperparameters are used to define the block structure and its behavior.

        in_channels : int
            The number of channels in the input samples.

        out_channels : int
            The number of channels in the output samples.

        kernel_size : tuple, optional
            The size of the convolutional kernel. Defaults to (3, 3).

        strides : tuple, optional
            The strides of the convolutional layer. Defaults to (1, 1).

        norm : bool, optional
            Whether to use normalization or not. Defaults to True.

        old_norm : bool, optional
            If True, batch normalization is used. Otherwise, spectral normalization is used. Defaults to False.

        lrelu : bool, optional
            Whether to use an activation layer or not. The activation function is taken from the hyperparams. Defaults
            to True.

        dropout : bool, optional
            Whether to use a dropout layer or not. The dropout probability is taken from the hyperparams. Defaults to
            True.
        """

        super(GAN_Discriminator_ConvBlock, self).__init__()


        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.strides      = strides
        self.norm         = norm
        self.old_norm     = old_norm
        self.lrelu        = lrelu
        self.dropout      = dropout


        layers = []


        if self.norm:

            if self.old_norm:
                layers.append(
                    nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.strides,
                              padding=(1, 1), padding_mode='replicate', bias=True),
                )
                layers.append(nn.BatchNorm2d(self.out_channels))

            else:
                layers.append(
                    nn.utils.parametrizations.spectral_norm(
                        nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.strides,
                                  padding=(1, 1), padding_mode='replicate', bias=True), dim=1
                    )
                )

        else:
            layers.append(
                nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, stride=self.strides, padding=(1, 1),
                          padding_mode='replicate', bias=True),
            )

        if self.lrelu:

            if hyperparams["activation"] == "elu":
                layers.append(nn.ELU(alpha=1.0, inplace=True))
            elif hyperparams["activation"] == "prelu":
                layers.append(nn.PReLU(num_parameters=self.out_channels, init=0.25))
            else:
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        if self.dropout:
            layers.append(nn.Dropout(p=hyperparams["p_dropout"]))


        self.layers = nn.Sequential(*layers)


    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        torch.Tensor
            The result of applying the block's different transformations over the input batch.
        """

        out = self.layers(tensor)

        return out


class GAN_Discriminator_LinearBlock(nn.Module):
    """Linear block for building networks in this module.
    
    A linear block contains a linear layer, an activation layer (optional), and a dropout layer (optional).

    Attributes
    ----------
    in_features : int
        The number of features in the input samples.

    out_features : int
        The number of features in the output samples.

    lrelu : bool
        Whether to use an activation layer or not.

    dropout : bool
        Whether to use a dropout layer or not.
    """

    def __init__(self, hyperparams, in_features, out_features, lrelu=True, dropout=True):
        """Defines and initializes the linear block.

        Parameters
        ----------
        hyperparams : dict
            Some hyperparameters are used to define the block structure and its behavior.

        in_features : int
            The number of features in the input samples.

        out_features : int
            The number of features in the output samples.

        lrelu : bool, optional
            Whether to use an activation layer or not. The activation function is taken from the hyperparams. Defaults
            to True.

        dropout : bool, optional
            Whether to use a dropout layer or not. The dropout probability is taken from the hyperparams. Defaults to
            True.
        """

        super(GAN_Discriminator_LinearBlock, self).__init__()


        self.in_features  = in_features
        self.out_features = out_features
        self.lrelu        = lrelu
        self.dropout      = dropout


        layers = []


        layers.append(
            nn.Linear(self.in_features, self.out_features),
        )

        if self.lrelu:

            if hyperparams["activation"] == "elu":
                layers.append(nn.ELU(alpha=1.0, inplace=True))
            elif hyperparams["activation"] == "prelu":
                layers.append(nn.PReLU(num_parameters=self.out_channels, init=0.25))
            else:
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        if self.dropout:
            layers.append(nn.Dropout(p=hyperparams["p_dropout"]))


        self.layers = nn.Sequential(*layers)


    def forward(self, tensor):
        """Processes a batch of samples.

        Parameters
        ----------
        tensor : torch.Tensor
            The input batch.

        Returns
        -------
        torch.Tensor
            The result of applying the block's different transformations over the input batch.
        """

        out = self.layers(tensor)

        return out


class GAN_Generator(nn.Module):
    """Generator for the GANs in this module.

    The size of the latent space is Z elements, and a total of 5 convolutional layers are applied to the input noise
    to generate the output sample.

    Furthermore, a spectral normalization layer is applied at each convolution, followed by an activation layer. In
    any case, the last convolution uses the tanh activation function to map the output sample to the [-1, 1] range.
    """

    def __init__(self, dataset, device, hyperparams):
        """Defines and initializes the network.

        Parameters
        ----------
        dataset : datasets.HyperDataset
            The properties of some layers depend on the target dataset.

        device : str
            The computational device that will be used to run the network.

        hyperparams : dict
            Some hyperparameters are used to define the network structure and its behavior.
        """

        super(GAN_Generator, self).__init__()


        self.bands         = dataset.bands
        self.patch_size    = dataset.patch_size
        self.classes_count = dataset.classes_count
        self.device        = device
        self.Z             = hyperparams["latent_size"]


        self.label_embedding = nn.Embedding(dataset.classes_count + 1, self.Z)

        # From 1x1 to 2x2
        self.conv_0 = nn.utils.parametrizations.spectral_norm(
            nn.ConvTranspose2d(self.Z, 16 * 8, (4, 4), stride=(2, 2), padding=(1,1), bias=True),
        )
        if hyperparams["activation"] == "elu":
            self.act_0 = nn.ELU(alpha=1.0, inplace=True)
        elif hyperparams["activation"] == "prelu":
            self.act_0 = nn.PReLU(num_parameters=16 * 8, init=0.25)
        else:
            self.act_0 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # To 4x4
        self.conv_1 = nn.utils.parametrizations.spectral_norm(
            nn.ConvTranspose2d(16 * 8, 16 * 4, (4, 4), stride=(2, 2), padding=(1,1), bias=True),
        )
        if hyperparams["activation"] == "elu":
            self.act_1 = nn.ELU(alpha=1.0, inplace=True)
        elif hyperparams["activation"] == "prelu":
            self.act_1 = nn.PReLU(num_parameters=16 * 4, init=0.25)
        else:
            self.act_1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # To 8x8
        self.conv_2 = nn.utils.parametrizations.spectral_norm(
            nn.ConvTranspose2d(16 * 4, 16 * 2, (4, 4), stride=(2, 2), padding=(1,1), bias=True)
        )
        if hyperparams["activation"] == "elu":
            self.act_2 = nn.ELU(alpha=1.0, inplace=True)
        elif hyperparams["activation"] == "prelu":
            self.act_2 = nn.PReLU(num_parameters=16 * 2, init=0.25)
        else:
            self.act_2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # To 16x16
        self.conv_3 = nn.utils.parametrizations.spectral_norm(
            nn.ConvTranspose2d(16 * 2, 16, (4, 4), stride=(2, 2), padding=(1,1), bias=True)
        )
        if hyperparams["activation"] == "elu":
            self.act_3 = nn.ELU(alpha=1.0, inplace=True)
        elif hyperparams["activation"] == "prelu":
            self.act_3 = nn.PReLU(num_parameters=16, init=0.25)
        else:
            self.act_3 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # To 32x32
        self.conv_4 = nn.utils.parametrizations.spectral_norm(
            nn.ConvTranspose2d(16, self.bands, (4, 4), stride=(2, 2), padding=(1,1), bias=True)
        )
        self.act_4 = nn.Tanh()


        self.apply(self.initialize_weights)


    def forward(self, tensors):
        """Processes a batch of samples.

        Parameters
        ----------
        tensors : tuple
            A tuple containing (1) the input samples and (2) the corresponding label for each one.

        Returns
        -------
        torch.Tensor
            The result of applying the network's different transformations over the input batch.

            More specifically, it generates fake samples from the input noise, with shape [batch_size, self.bands,
            self.patch_size, self.patch_size], conditioned on the desired labels.
        """

        if isinstance(tensors, tuple):
            noise, label = tensors
        else:
            # nasty fix to be able to run torchsummary and tensorboard
            noise, label = tensors, torch.zeros((2), dtype=torch.int64, device=self.device)


        # The label embedding is incorporated into the noise via an element-wise multiplication
        z = noise * self.label_embedding(label)

        # We need to turn the noise into a 4D tensor with shape [batch_size, Z, 1, 1]
        z = z.view(-1, self.Z, 1, 1)


        image = self.conv_0(z)
        image = self.act_0(image)

        image = self.conv_1(image)
        image = self.act_1(image)

        image = self.conv_2(image)
        image = self.act_2(image)

        image = self.conv_3(image)
        image = self.act_3(image)

        image = self.conv_4(image)
        image = self.act_4(image)


        return image


    @staticmethod
    def initialize_weights(child):
        """Receives a layer and initializes its weights and biases if needed.

        Parameters
        ----------
        child : torch.nn.Module
            The layer.
        """

        if isinstance(child, nn.ConvTranspose2d):
            nn.init.xavier_normal_(child.weight)
            nn.init.constant_(child.bias, 0)


    def __str__(self):         

        return "[*] GAN_Generator network summary:\n" \
                f"{torchsummary.summary(self, [self.Z], verbose=0, device=self.device)}"


def dual_train_ACGAN(networks, data_loader, dataset, hyperparams):
    """Trains the given ACGAN on a certain dataset, using its train set.
    
    After each epoch, the dataset's validation set is also used to monitor under/overfitting. Every 5 epochs, the
    generator's output on a fixed set of noise is also saved to a file.

    Parameters
    ----------
    networks : torch.nn.Module
        The ACGAN network.

    data_loader : torch.utils.data.DataLoader
        A PyTorch DataLoader that wraps the target datasets.HyperDataset.

    dataset : datasets.HyperDataset
        The target dataset.

    hyperparams : dict
        Some hyperparameters are used to control the behavior of the network.
    """

    discriminator = networks.discriminator
    generator     = networks.generator


    # One entry per epoch
    losses_d         = []
    losses_d_real    = []
    losses_d_fake    = []
    losses_g         = []
    train_accuracies = []
    val_accuracies   = []

    fixed_samples_per_class = 10

    fixed_noise = torch.randn(
        (fixed_samples_per_class * dataset.classes_count, hyperparams["latent_size"]), dtype=torch.float32,
        device=networks.device
    )

    fixed_labels = torch.repeat_interleave(
        torch.arange(dataset.classes_count, dtype=torch.int64, device=networks.device), fixed_samples_per_class
    )


    for epoch in tqdm.tqdm(range(hyperparams["epochs"]), total=hyperparams["epochs"],
                        desc="[*] Training the networks...", file=sys.stdout):

        # One entry per batch
        epoch_losses_d      = []
        epoch_losses_d_real = []
        epoch_losses_d_fake = []
        epoch_losses_g      = []


        # In case that any of these modes get changed during the previous epoch
        # eval() instructs normalization layers to use running statistics instead of batch statistics, which helps
        # in training
        networks.eval()
        dataset.to_train()


        for batch_id, (samples, targets, _) in enumerate(data_loader):

            # 1. Train the discriminator on real samples

            lies_real                  = torch.ones((samples.shape[0], 1), dtype=torch.float32, device=networks.device)
            real_samples, real_targets = samples.to(networks.device, non_blocking=True), targets.to(networks.device,
                                                    non_blocking=True)

            # Gradients from the previous batch need to be cleared, as PyTorch accumulates them if not told otherwise
            networks.optimizer_D.zero_grad(set_to_none=True)
            networks.optimizer_G.zero_grad(set_to_none=True)

            # Runs all real samples through the network
            D_output_real, _, D_output_real_lies = discriminator(real_samples)

            # By how much the network has missed the correct predictions
            D_loss_real = (networks.classifier(D_output_real, real_targets) +
                           networks.loss(D_output_real_lies, lies_real)) / 2


            # 2. Train the discriminator on fake samples

            # On each batch, 1 / (classes_count + 1) of the samples are fake
            batch_size = real_samples.shape[0]
            fake_samples_count = max(batch_size // dataset.classes_count, 1)

            # No conditional noise distribution is taken from the generator
            noise = torch.randn(
                (fake_samples_count, hyperparams["latent_size"]), dtype=torch.float32, device=networks.device
            )

            random_labels = torch.randint(
                0, dataset.classes_count, (fake_samples_count,), dtype=torch.int64, device=networks.device
            )

            lies_fake = torch.zeros(
                (fake_samples_count, 1), dtype=torch.float32, device=networks.device
            )

            # Runs all fake samples through the networks
            fake_samples                         = generator((noise, random_labels))
            D_output_fake, _, D_output_fake_lies = discriminator(fake_samples)

            # By how much the network has missed the correct predictions
            D_loss_fake = (networks.classifier(D_output_fake, random_labels) +
                           networks.loss(D_output_fake_lies, lies_fake)) / 2

            # Computes the corresponding gradient for each component of the network
            D_loss = D_loss_real + D_loss_fake
            D_loss.backward()

            # And adjusts all weights & biases using those gradients
            networks.optimizer_D.step()

            # Some metrics
            epoch_losses_d.append(D_loss.item())
            epoch_losses_d_real.append(D_loss_real.item())
            epoch_losses_d_fake.append(D_loss_fake.item())            


            # 3. Train the generator

            # Gradients from the previous batch need to be cleared, as PyTorch accumulates them if not told otherwise
            networks.optimizer_D.zero_grad(set_to_none=True)
            networks.optimizer_G.zero_grad(set_to_none=True)

            # The generator receives as many samples as the batch size

            lies_real = torch.ones((batch_size + fake_samples_count, 1), dtype=torch.float32, device=networks.device)

            # No conditional noise distribution is taken from the generator
            noise = torch.randn(
                (batch_size + fake_samples_count, hyperparams["latent_size"]), dtype=torch.float32,
                device=networks.device
            )

            random_labels = torch.randint(
                0, dataset.classes_count, (batch_size + fake_samples_count,), dtype=torch.int64, device=networks.device
            )

            # Runs all fake samples through the networks
            fake_samples                         = generator((noise, random_labels))
            D_output_fake, _, D_output_fake_lies = discriminator(fake_samples)

            # By how much the network has missed the correct predictions
            G_loss = (networks.classifier(D_output_fake, random_labels) +
                      networks.loss(D_output_fake_lies, lies_real)) / 2

            # Computes the corresponding gradient for each component of the network
            G_loss.backward()

            # And adjusts all weights & biases using those gradients
            networks.optimizer_G.step()

            # Some metrics
            epoch_losses_g.append(G_loss.item())


            # Check how the generator is doing by saving G's output on fixed_noise
            #
            # - Every 5 epochs
            # - And in the last epoch
            if ((epoch % 5 == 0) or (epoch == hyperparams["epochs"] - 1)) and (batch_id == len(data_loader) - 1):

                networks.eval()

                with torch.no_grad():

                    fake_samples = generator((fixed_noise, fixed_labels))
                    fake_samples_grid = torchvision.utils.make_grid(
                        fake_samples[:, 0:3, :, :], nrow=fixed_samples_per_class, normalize=True
                    )

                    path = f"logs/{dataset.name_no_slashes}_{dataset.segmented}_{dataset.ratios[0]}-{dataset.ratios[1]}_ACGAN-fake-samples_e-{epoch}.png"
                    torchvision.utils.save_image(fake_samples_grid, path)

                networks.eval()


            # Some clean-up before the next batch
            del(real_samples, real_targets, D_output_real, D_output_fake, D_loss_real, D_loss_fake, D_loss, G_loss, \
                fake_samples, noise, random_labels, lies_real, D_output_real_lies, lies_fake, D_output_fake_lies)


        # The epoch's total losses can be computed once all batches are fed
        losses_d.append(np.mean(epoch_losses_d))
        losses_d_real.append(np.mean(epoch_losses_d_real))
        losses_d_fake.append(np.mean(epoch_losses_d_fake))
        losses_g.append(np.mean(epoch_losses_g))


        # The validation function can also be run now
        val_accuracies.append(validate(networks.discriminator, data_loader, dataset, hyperparams))
        networks.eval()
        dataset.to_train()


        # A tiny report of the training process is displayed after every epoch
        tqdm.tqdm.write(
            f"[*] Training epoch: {epoch + 1}/{hyperparams['epochs']}:\n" \
            f"\tDiscriminator's mean loss (average): {losses_d[-1]}\n" \
            f"\tDiscriminator's mean loss (real images): {losses_d_real[-1]}\n" \
            f"\tDiscriminator's mean loss (fake images): {losses_d_fake[-1]}\n" \
            f"\tGenerators's mean loss: {losses_g[-1]}\n" \
            f"\tValidation accuracy: {val_accuracies[-1] * 100} %\n" \
        )


    print("[*] Training finished!")


def dual_train_BAGAN(networks, data_loader, dataset, hyperparams):
    """Trains the given BAGAN on a certain dataset, using its train set.

    The training is performed in two phases:

    - First, the autoencoder is trained on the train set.
    - Then, all parameters in the autoencoder are transferred to the GAN network, which is now trained.
    
    While training the GAN network:
    
    - After each epoch, the dataset's validation set is also used to monitor under/overfitting.
    - Every 5 epochs, the generator's output on a fixed set of noise is also saved to a file.

    Parameters
    ----------
    networks : torch.nn.Module
        The BAGAN network.

    data_loader : torch.utils.data.DataLoader
        A PyTorch DataLoader that wraps the target datasets.HyperDataset.

    dataset : datasets.HyperDataset
        The target dataset.

    hyperparams : dict
        Some hyperparameters are used to control the behavior of the network.
    """

    autoencoder   = networks.autoencoder
    discriminator = networks.discriminator
    generator     = networks.generator


    # 1st step: train the autoencoder

    # One entry per epoch
    losses = []


    for epoch in tqdm.tqdm(range(hyperparams["epochs"]), total=hyperparams["epochs"],
                           desc="[*] Training the autoencoder...", file=sys.stdout):

        # One entry per batch
        epoch_losses = []


        # In case that any of these modes get changed during the previous epoch
        autoencoder.train()
        dataset.to_train()


        for batch_id, (samples, targets, _) in enumerate(data_loader):

            samples, targets = samples.to(networks.device, non_blocking=True), \
                               targets.to(networks.device, non_blocking=True)

            # Gradients from the previous batch need to be cleared, as PyTorch accumulates them if not told otherwise
            networks.optimizer_A.zero_grad(set_to_none=True)
            
            # Runs all samples through the network
            output = autoencoder((samples, targets))
                
            # How similar are the reconstructed samples to the original ones
            loss = networks.criterion_A(output, samples)

            # Computes the corresponding gradient for each component of the network
            loss.backward()

            # And adjusts all weights & biases using those gradients
            networks.optimizer_A.step()


            # Some metrics from the current batch
            epoch_losses.append(loss.item())


            # Some clean-up before the next batch
            del(samples, targets, output, loss)


        # The epoch's total loss can be computed once all batches are fed
        losses.append(np.mean(epoch_losses))


        # A tiny report of the training process is displayed along every epoch
        tqdm.tqdm.write(
            f"[*] Training epoch: {epoch + 1}/{hyperparams['epochs']}:\n" \
            f"\tMean loss: {losses[-1]}\n"
        )


    print("[*] Training of the autoencoder finished!")


    # 2nd step: transfer the weights from the autoencoder to the generator and the discriminator
    
    networks.discriminator.load_state_dict(autoencoder.discriminator.state_dict())
    networks.generator.load_state_dict(autoencoder.generator.state_dict())


    # 3rd step: train the GAN

    # One entry per epoch
    losses_d         = []
    losses_d_real    = []
    losses_d_fake    = []
    losses_g         = []
    train_accuracies = []
    val_accuracies   = []

    fixed_samples_per_class = 10

    fixed_noise = torch.randn(
        (fixed_samples_per_class * dataset.classes_count, hyperparams["latent_size"]), dtype=torch.float32,
        device=networks.device
    )

    fixed_labels = torch.repeat_interleave(
        torch.arange(dataset.classes_count, dtype=torch.int64, device=networks.device), fixed_samples_per_class
    )


    for epoch in tqdm.tqdm(range(hyperparams["epochs"]), total=hyperparams["epochs"],
                        desc="[*] Training the networks...", file=sys.stdout):

        # One entry per batch
        epoch_losses_d      = []
        epoch_losses_d_real = []
        epoch_losses_d_fake = []
        epoch_losses_g      = []


        # In case that any of these modes get changed during the previous epoch
        # eval() instructs normalization layers to use running statistics instead of batch statistics, which helps
        # in training
        networks.eval()
        dataset.to_train()


        for batch_id, (samples, targets, _) in enumerate(data_loader):

            # 1. Train the discriminator on real samples

            real_samples, real_targets = samples.to(networks.device, non_blocking=True), \
                                         targets.to(networks.device, non_blocking=True)

            # Gradients from the previous batch need to be cleared, as PyTorch accumulates them if not told otherwise
            networks.optimizer_D.zero_grad(set_to_none=True)
            networks.optimizer_G.zero_grad(set_to_none=True)

            # Runs all real samples through the network
            D_output_real, _ = discriminator(real_samples)

            # By how much the network has missed the correct predictions
            D_loss_real = networks.classifier(D_output_real, real_targets)


            # 2. Train the discriminator on fake samples

            # On each batch, 1 / (classes_count + 1) of the samples are fake
            batch_size = real_samples.shape[0]
            fake_samples_count = max(batch_size // dataset.classes_count, 1)

            # No conditional noise distribution is taken from the generator
            noise = torch.randn(
                (fake_samples_count, hyperparams["latent_size"]), dtype=torch.float32, device=networks.device
            )

            fake_targets = torch.full(
                (fake_samples_count,), dataset.classes_count, dtype=torch.int64, device=networks.device
            )

            random_labels = torch.randint(
                0, dataset.classes_count, (fake_samples_count,), dtype=torch.int64, device=networks.device
            )

            # Runs all fake samples through the networks
            fake_samples     = generator((noise, random_labels))
            D_output_fake, _ = discriminator(fake_samples)

            # By how much the network has missed the correct predictions
            D_loss_fake = networks.classifier(D_output_fake, fake_targets)

            # Computes the corresponding gradient for each component of the network
            D_loss = D_loss_real + D_loss_fake
            D_loss.backward()

            # And adjusts all weights & biases using those gradients
            networks.optimizer_D.step()

            # Some metrics
            epoch_losses_d.append(D_loss.item())
            epoch_losses_d_real.append(D_loss_real.item())
            epoch_losses_d_fake.append(D_loss_fake.item())            


            # 3. Train the generator

            # Gradients from the previous batch need to be cleared, as PyTorch accumulates them if not told otherwise
            networks.optimizer_D.zero_grad(set_to_none=True)
            networks.optimizer_G.zero_grad(set_to_none=True)

            # The generator receives as many samples as the batch size

            # No conditional noise distribution is taken from the generator
            noise = torch.randn(
                (batch_size + fake_samples_count, hyperparams["latent_size"]), dtype=torch.float32,
                device=networks.device
            )

            random_labels = torch.randint(
                0, dataset.classes_count, (batch_size + fake_samples_count,), dtype=torch.int64, device=networks.device
            )

            # Runs all fake samples through the networks
            fake_samples     = generator((noise, random_labels))
            D_output_fake, _ = discriminator(fake_samples)

            # By how much the network has missed the correct predictions
            G_loss = networks.classifier(D_output_fake, random_labels)

            # Computes the corresponding gradient for each component of the network
            G_loss.backward()

            # And adjusts all weights & biases using those gradients
            networks.optimizer_G.step()

            # Some metrics
            epoch_losses_g.append(G_loss.item())


            # Check how the generator is doing by saving G's output on fixed_noise
            #
            # - Every 5 epochs
            # - And in the last epoch
            if ((epoch % 5 == 0) or (epoch == hyperparams["epochs"] - 1)) and (batch_id == len(data_loader) - 1):

                networks.eval()

                with torch.no_grad():

                    fake_samples = generator((fixed_noise, fixed_labels))
                    fake_samples_grid = torchvision.utils.make_grid(
                        fake_samples[:, 0:3, :, :], nrow=fixed_samples_per_class, normalize=True
                    )

                    path = f"logs/{dataset.name_no_slashes}_{dataset.segmented}_{dataset.ratios[0]}-{dataset.ratios[1]}_ResBaGAN-fake-samples_e-{epoch}.png"
                    torchvision.utils.save_image(fake_samples_grid, path)

                networks.eval()


            # Some clean-up before the next batch
            del(real_samples, real_targets, D_output_real, D_output_fake, D_loss_real, D_loss_fake, D_loss, G_loss, \
                fake_samples, fake_targets, noise, random_labels)


        # The epoch's total losses can be computed once all batches are fed
        losses_d.append(np.mean(epoch_losses_d))
        losses_d_real.append(np.mean(epoch_losses_d_real))
        losses_d_fake.append(np.mean(epoch_losses_d_fake))
        losses_g.append(np.mean(epoch_losses_g))
        

        # The validation function can also be run now
        val_accuracies.append(validate(networks.discriminator, data_loader, dataset, hyperparams))
        networks.eval()
        dataset.to_train()
        

        # A tiny report of the training process is displayed after every epoch
        tqdm.tqdm.write(
            f"[*] Training epoch: {epoch + 1}/{hyperparams['epochs']}:\n" \
            f"\tDiscriminator's mean loss (average): {losses_d[-1]}\n" \
            f"\tDiscriminator's mean loss (real images): {losses_d_real[-1]}\n" \
            f"\tDiscriminator's mean loss (fake images): {losses_d_fake[-1]}\n" \
            f"\tGenerators's mean loss: {losses_g[-1]}\n" \
            f"\tValidation accuracy: {val_accuracies[-1] * 100} %\n" \
        )


    print("[*] Training finished!")


def train(network, data_loader, dataset, hyperparams):
    """Trains the given CNN on a certain dataset, using its train set.
    
    After each epoch, the dataset's validation set is also used to monitor under/overfitting.

    Parameters
    ----------
    network : torch.nn.Module
        The CNN network.

    data_loader : torch.utils.data.DataLoader
        A PyTorch DataLoader that wraps the target datasets.HyperDataset.

    dataset : datasets.HyperDataset
        The target dataset.

    hyperparams : dict
        Some hyperparameters are used to control the behavior of the network.
    """

    # One entry per epoch
    losses           = []
    train_accuracies = []
    val_accuracies   = []


    for epoch in tqdm.tqdm(range(hyperparams["epochs"]), total=hyperparams["epochs"],
                           desc="[*] Training the network...", file=sys.stdout):

        # One entry per batch
        epoch_losses = []


        # In case that any of these modes get changed during the previous epoch
        network.train()
        dataset.to_train()


        for batch_id, (samples, targets, _) in enumerate(data_loader):

            samples, targets = samples.to(network.device, non_blocking=True), \
                               targets.to(network.device, non_blocking=True)

            # Gradients from the previous batch need to be cleared, as PyTorch accumulates them if not told otherwise
            network.optimizer.zero_grad(set_to_none=True)
            
            # Runs all samples through the network
            output, _ = network(samples)
                
            # By how much the network has missed the correct predictions
            loss = network.classifier(output, targets)

            # Computes the corresponding gradient for each component of the network
            loss.backward()

            # And adjusts all weights & biases using those gradients
            network.optimizer.step()


            # Some metrics from the current batch
            epoch_losses.append(loss.item())


            # Some clean-up before the next batch
            del(samples, targets, output, loss)


        # The epoch's total loss can be computed once all batches are fed
        losses.append(np.mean(epoch_losses))


        # The validation procedure can also be run now
        val_accuracies.append(validate(network, data_loader, dataset, hyperparams))
        network.train()
        dataset.to_train()


        # A tiny report of the training process is displayed along every epoch
        tqdm.tqdm.write(
            f"[*] Training epoch: {epoch + 1}/{hyperparams['epochs']}:\n" \
            f"\tMean loss: {losses[-1]}\n" \
            f"\tValidation accuracy: {val_accuracies[-1] * 100} %\n"
        )


    print("[*] Training finished!")


def validate(network, data_loader, dataset, hyperparams):
    """Measures the accuracy of the given classifier network on the validation set of a certain dataset.

    More specifically, this function runs the network on the validation set and compares its predictions with the
    ground truth labels. The accuracy is then computed as the ratio between the number of correct predictions and
    the total number of samples.

    Parameters
    ----------
    network : torch.nn.Module
        The classifier network.

    data_loader : torch.utils.data.DataLoader
        A PyTorch DataLoader that wraps the target datasets.HyperDataset.

    dataset : datasets.HyperDataset
        The target dataset.

    hyperparams : dict
        Some hyperparameters are used to control the behavior of the network.
    """

    right_guesses = 0
    total_guesses = 0


    network.eval()
    dataset.to_validation()


    for batch_id, (samples, targets, _) in enumerate(data_loader):

        # Gradients are only needed when training a network
        with torch.no_grad():

            samples, targets = samples.to(network.device, non_blocking=True), \
                               targets.to(network.device, non_blocking=True)

            # Runs all samples through the network
            # The resulting tensor contains, for each sample, the computed probability of it belonging to each class
            if isinstance(network, CNN2D) or isinstance(network, CNN2D_Residual):
                output, _                        = network(samples)
            elif isinstance(network, ACGAN_Discriminator) or isinstance(network, ResACGAN_Discriminator):
                output, _, _                     = network(samples)
            elif isinstance(network, BAGAN_Discriminator) or isinstance(network, ResBaGAN_Discriminator):
                output, _                        = network(samples)
                output[:, dataset.classes_count] = -math.inf # Fake class is disabled when testing
            else:
                raise NotImplementedError("Unknown network class!")

            # We need to retrieve manually the class predictions
            try:
                _, output = torch.max(output, dim=1)
            # Some nets may return a 1D tensor when batch_size = 1
            except IndexError:
                _, output = torch.max(output, dim=0)

            for out, tar in zip(output.view(-1), targets.view(-1)):
                right_guesses += out.item() == tar.item()
                total_guesses += 1

            # Some clean-up before the next batch
            del (samples, targets, output)


    return right_guesses / total_guesses


def test(network, data_loader, dataset, hyperparams, multiple_labels=False, full_dataset=False):
    """Measures the accuracy of the given classifier network on the test set of a certain dataset.

    More specifically, this function runs the network on the test set and compares its predictions with the ground
    truth labels. The accuracy is then computed in terms of OA, AA, and kappa.

    Parameters
    ----------
    network : torch.nn.Module
        The classifier network.

    data_loader : torch.utils.data.DataLoader
        A PyTorch DataLoader that wraps the target datasets.HyperDataset.

    dataset : datasets.HyperDataset
        The target dataset.

    hyperparams : dict
        Some hyperparameters are used to control the behavior of the network.

    multiple_labels : bool, optional
        If True, each prediction is compared against an array of ground truth labels instead of a single value. This
        functionality is intended for computing pixel-level accuracy on segmented datasets. Defaults to False.

    full_dataset : bool, optional
        If True, the entire dataset is used for testing. Otherwise, only the test set is used. Defaults to False.
    """

    # We need to store each prediction in order to compute at the end the Cohen's Kappa coefficient
    all_outputs = []
    all_targets = []

    # We add an extra "fake" class that is used by GANs 
    confusion_matrix = np.zeros((network.classes_count + 1, network.classes_count + 1), dtype=np.int32)


    print("[*] Testing the network...")

    network.eval()
    dataset.to_test()
    dataset.set_pixel_level_labels(multiple_labels)


    for batch_id, (samples, targets, targets_pixel_level) in enumerate(data_loader):

        # Gradients are only needed when training a network
        with torch.no_grad():

            samples, targets = samples.to(network.device, non_blocking=True), \
                               targets.to(network.device, non_blocking=True)

            # Runs all samples through the network
            # The resulting tensor contains, for each sample, the computed probability of it belonging to each class
            if isinstance(network, CNN2D) or isinstance(network, CNN2D_Residual):
                output, _                        = network(samples)
            elif isinstance(network, ACGAN_Discriminator) or isinstance(network, ResACGAN_Discriminator):
                output, _, _                     = network(samples)
            elif isinstance(network, BAGAN_Discriminator) or isinstance(network, ResBaGAN_Discriminator):
                output, _                        = network(samples)
                output[:, dataset.classes_count] = -math.inf # Fake class is disabled when testing
            else:
                raise NotImplementedError("Unknown network class!")

            # We need to retrieve manually the class predictions
            try:
                _, output = torch.max(output, dim=1)
            # Some nets may return a 1D tensor when batch_size = 1
            except IndexError:
                _, output = torch.max(output, dim=0)


            if not multiple_labels:

                for out, tar in zip(output.view(-1), targets.view(-1)):

                    all_outputs.append(out.item())
                    all_targets.append(tar.item())
                    confusion_matrix[tar.item(), out.item()] += 1

            else:

                for out, tars in zip(output.view(-1), targets_pixel_level):
                    for tar in tars:

                        if tar == -1:  # -1 is a sentinel value to point out that there are no
                            break      # more labels to process
                        
                        all_outputs.append(out.item())
                        all_targets.append(tar.item())
                        confusion_matrix[tar.item(), out.item()] += 1


            # Some clean-up before the next batch
            del (samples, targets, output)


    ############################ For compatibility with https://doi.org/10.3390/rs13142687 ############################
    if full_dataset:


        # Validation set
        dataset.to_validation()

        for batch_id, (samples, targets, targets_pixel_level) in enumerate(data_loader):

            # Gradients are only needed when training a network
            with torch.no_grad():

                samples, targets = samples.to(network.device, non_blocking=True), \
                                   targets.to(network.device, non_blocking=True)

                # Runs all samples through the network
                # The resulting tensor contains, for each sample, the computed probability of it belonging to each
                # class
                if isinstance(network, CNN2D) or isinstance(network, CNN2D_Residual):
                    output, _                        = network(samples)
                elif isinstance(network, ACGAN_Discriminator) or isinstance(network, ResACGAN_Discriminator):
                    output, _, _                     = network(samples)
                elif isinstance(network, BAGAN_Discriminator) or isinstance(network, ResBaGAN_Discriminator):
                    output, _                        = network(samples)
                    output[:, dataset.classes_count] = -math.inf # Fake class is disabled when testing
                else:
                    raise NotImplementedError("Unknown network class!")

                # We need to retrieve manually the class predictions
                try:
                    _, output = torch.max(output, dim=1)
                # Some nets may return a 1D tensor when batch_size = 1
                except IndexError:
                    _, output = torch.max(output, dim=0)


                if not multiple_labels:

                    for out, tar in zip(output.view(-1), targets.view(-1)):

                        all_outputs.append(out.item())
                        all_targets.append(tar.item())
                        confusion_matrix[tar.item(), out.item()] += 1

                else:

                    for out, tars in zip(output.view(-1), targets_pixel_level):
                        for tar in tars:

                            if tar == -1:  # -1 is a sentinel value to point out that there are no
                                break      # more labels to process
                            
                            all_outputs.append(out.item())
                            all_targets.append(tar.item())
                            confusion_matrix[tar.item(), out.item()] += 1


                # Some clean-up before the next batch
                del (samples, targets, output)


        # Train set
        dataset.to_train()

        for batch_id, (samples, targets, targets_pixel_level) in enumerate(data_loader):

            # Gradients are only needed when training a network
            with torch.no_grad():

                samples, targets = samples.to(network.device, non_blocking=True), \
                                   targets.to(network.device, non_blocking=True)

                # Runs all samples through the network
                # The resulting tensor contains, for each sample, the computed probability of it belonging to each
                # class
                if isinstance(network, CNN2D) or isinstance(network, CNN2D_Residual):
                    output, _                        = network(samples)
                elif isinstance(network, ACGAN_Discriminator) or isinstance(network, ResACGAN_Discriminator):
                    output, _, _                     = network(samples)
                elif isinstance(network, BAGAN_Discriminator) or isinstance(network, ResBaGAN_Discriminator):
                    output, _                        = network(samples)
                    output[:, dataset.classes_count] = -math.inf # Fake class is disabled when testing
                else:
                    raise NotImplementedError("Unknown network class!")

                # We need to retrieve manually the class predictions
                try:
                    _, output = torch.max(output, dim=1)
                # Some nets may return a 1D tensor when batch_size = 1
                except IndexError:
                    _, output = torch.max(output, dim=0)


                if not multiple_labels:

                    for out, tar in zip(output.view(-1), targets.view(-1)):

                        all_outputs.append(out.item())
                        all_targets.append(tar.item())
                        confusion_matrix[tar.item(), out.item()] += 1

                else:

                    for out, tars in zip(output.view(-1), targets_pixel_level):
                        for tar in tars:

                            if tar == -1:  # -1 is a sentinel value to point out that there are no
                                break      # more labels to process
                            
                            all_outputs.append(out.item())
                            all_targets.append(tar.item())
                            confusion_matrix[tar.item(), out.item()] += 1


                # Some clean-up before the next batch
                del (samples, targets, output)
    ###################################################################################################################


    # The final metrics

    right_guesses    = np.trace(confusion_matrix)              # Adds all elements in the diagonal
    total_guesses    = np.concatenate(confusion_matrix).sum()  # Adds all elements in the matrix
    
    overall_accuracy = right_guesses / total_guesses


    usable_classes   = 0
    average_accuracy = 0

    for class_i in range(network.classes_count + 1):
        # In case any class has no samples in the dataset
        if sum(confusion_matrix[class_i]) > 0:

            usable_classes   += 1
            average_accuracy += confusion_matrix[class_i][class_i] / sum(confusion_matrix[class_i])

    average_accuracy /= usable_classes


    kappa = sklearn.metrics.cohen_kappa_score(all_outputs, all_targets)


    # pandas is used to print the confusion matrix in a nice way
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.expand_frame_repr", False):

        frame = pd.DataFrame(confusion_matrix)

        classes = dataset.classes.copy()
        classes.append("fake")

        # Names for the frame's cols & rows
        frame.columns, frame.index = classes, classes

        print(
            f"[acc] Overall Accuracy (OA): {overall_accuracy * 100} %\n"
            f"[acc] Average Accuracy (AA): {average_accuracy * 100} %\n"
            f"[acc] Cohen's Kappa (k): {kappa * 100} %\n"
            f"[acc] Confusion matrix:\n"
            f"{frame}"
        )
