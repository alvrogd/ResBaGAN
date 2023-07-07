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


"""Tests the synthesis performance of a pretrained GAN network on a given dataset.

More specifically, this script:

- Loads the specified pretrained GAN model from disk.
- Loads the residual classifier pretrained on the same dataset to use it as reference.
- Evaluates the synthesis performance of the generator in the GAN in terms of FID score.

The FID score is evaluated in a per-class basis, by retrieving all available samples for a certain class, generating
the same number of fake samples, and comparing the two feature distributions that the reference network extracts from
the real and fake samples.
"""


__author__ = "alvrogd"


import argparse

import numpy as np
import scipy
import torch

import datasets
import networks


parser = argparse.ArgumentParser(
    prog="test_synthesis.py",
    description="Tests the synthesis performance of a pretrained GAN network on a given dataset"
)

parser.add_argument(
    "--dataset_path",
    type=str,
    action="store",
    # As saved by preprocess_dataset.py
    default="preprocessed/hyperdataset"
)

# Most of the following arguments are not used in this script, but are required to instantiate the networks as in
# run_network.py
parser.add_argument(
    "--data_augmentation",
    type=int,
    action="store",
    # 0: no data augmentation, > 0: data augmentation
    default=0
)

parser.add_argument(
    "--network",
    type=str,
    action="store",
    default="ResBaGAN"
)

parser.add_argument(
    "--latent_size",
    type=int,
    action="store",
    default=128
)

parser.add_argument(
    "--activation",
    type=str,
    action="store",
    default="lrelu"
)

parser.add_argument(
    "--p_dropout",
    type=float,
    action="store",
    default=0.05
)

parser.add_argument(
    "--weight_init",
    type=str,
    action="store",
    default="xavier"
)

parser.add_argument(
    "--learning_rate",
    type=float,
    action="store",
    default=0.001
)

parser.add_argument(
    "--epochs",
    type=int,
    action="store",
    default=600
)

parser.add_argument(
    "--batch_size",
    type=int,
    action="store",
    default=32
)

parser.add_argument(
    "--num_workers",
    type=int,
    action="store",
    # To speed-up dataloaders
    default=4
)

parser.add_argument(
    "--device",
    type=str,
    action="store",
    # A CUDA-compatible GPU will be automatically used if available
    default=f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
)

parser.add_argument(
    "--model_path",
    type=str,
    action="store",
    default="logs/ResBaGAN_model.pt"
)

parser.add_argument(
    "--reference_model_path",
    type=str,
    action="store",
    default="CNN2D_Residual_model.pt"
)

args = parser.parse_args()
print(f"[*] Arguments: {vars(args)}")


hyperparams = {
    "latent_size":   args.latent_size,
    "activation":    args.activation,
    "p_dropout":     args.p_dropout,
    "weight_init":   args.weight_init,
    "learning_rate": args.learning_rate,
    "epochs":        args.epochs,
    "batch_size":    args.batch_size,
    "num_workers":   args.num_workers,
    "device":        args.device,
}


# cuDNN must be enabled manually if there is a CUDA-compatible GPU available
if hyperparams["device"] != "cpu":
    # The Docker image has cuDNN installed
    torch.backends.cudnn.enabled = True
    # Therefore, we will tell cuDNN to search for the fastest algorithms before training to get
    # the most out of the GPU
    torch.backends.cudnn.benchmark = True


# Loading from disk the preprocessed dataset that will be used to test the network
dataset = datasets.read_preprocessed_dataset(args.dataset_path)
dataset.set_data_augmentation(False)
print(dataset)

# The custom HyperDataset object contains all the train, validation and test data
#   --> But it will wrapped into a PyTorch data feeder for convenience
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=hyperparams["batch_size"],
    shuffle=True,
    num_workers=hyperparams["num_workers"],
    pin_memory=hyperparams["device"] != "cpu",
)


# Building the requested and reference networks
#   --> The reference network is used to compute synthethic distributions and per-class FID score

if args.network == "CNN2D" or args.network == "CNN2D_Residual":
    raise ValueError("CNN2D and CNN2D_Residual networks are not GANs")
elif args.network == "ACGAN":
    network = networks.ACGAN(dataset, hyperparams["device"], hyperparams)
elif args.network == "ResACGAN":
    network = networks.ResACGAN(dataset, hyperparams["device"], hyperparams)
elif args.network == "BAGAN":
    network = networks.BAGAN(dataset, hyperparams["device"], hyperparams)
elif args.network == "ResBaGAN":
    network = networks.ResBaGAN(dataset, hyperparams["device"], hyperparams)
else:
    raise ValueError(f"[!] Unknown network: {args.network}")

print(network)

reference_network = networks.CNN2D_Residual(dataset, hyperparams["device"], hyperparams)
print(reference_network)


# Loading the pretrained models

network.load_state_dict(torch.load(args.model_path, map_location=hyperparams["device"]))
network.eval()

reference_network.load_state_dict(torch.load(args.reference_model_path, map_location=hyperparams["device"]))
reference_network.eval()

discriminator = reference_network
generator     = network.generator


# Now we will test the synthesis performance of the requested network by computing the FID score for each class
for class_i in range(dataset.classes_count):

    print(f"[*] Computing FID score for [C{class_i}] {dataset.classes[class_i]}")


    # First, we need to gather all real samples for the current class

    all_samples = []


    dataset.to_train()

    for batch_id, (samples, targets, _) in enumerate(data_loader):
        for sample, target in zip(samples, targets):
            if target.item() == class_i:
                all_samples.append(sample.numpy())

        # Some clean-up before the next batch
        del(samples, targets)


    dataset.to_validation()

    for batch_id, (samples, targets, _) in enumerate(data_loader):
        for sample, target in zip(samples, targets):
            if target.item() == class_i:
                all_samples.append(sample.numpy())

        # Some clean-up before the next batch
        del(samples, targets)


    dataset.to_test()

    for batch_id, (samples, targets, _) in enumerate(data_loader):
        for sample, target in zip(samples, targets):
            if target.item() == class_i:
                all_samples.append(sample.numpy())

        # Some clean-up before the next batch
        del(samples, targets)


    all_samples = np.asarray(np.copy(all_samples), dtype=np.float32)
    all_samples = torch.from_numpy(all_samples)

    print(f"\t{all_samples.shape[0]} samples gathered")


    ###################################################################################################################
    ##### Code adapted from https://github.com/mseitzer/pytorch-fid/blob/4dfcdc2b70217883da8b8867c5cf8db7ddfffcda/src/pytorch_fid/fid_score.py
    ##### to just use the relevant parts
    #####
    ##### Copyright 2018 Institute of Bioinformatics, JKU Linz
    #####
    ##### Licensed under the Apache License, Version 2.0 (the "License");
    ##### you may not use this file except in compliance with the License.
    ##### You may obtain a copy of the License at

    ##### http://www.apache.org/licenses/LICENSE-2.0

    ##### Unless required by applicable law or agreed to in writing, software
    ##### distributed under the License is distributed on an "AS IS" BASIS,
    ##### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    ##### See the License for the specific language governing permissions and
    ##### limitations under the License.

    act1 = np.empty((all_samples.shape[0], 128), dtype=np.float32)
    act2 = np.empty((all_samples.shape[0], 128), dtype=np.float32)

    with torch.no_grad():
        
        all_samples = all_samples.to(network.device)
        _, features = discriminator(all_samples)
        act1        = features.cpu().numpy()
        
        noise = torch.randn(
            (all_samples.shape[0], hyperparams["latent_size"]), dtype=torch.float32, device=network.device
        )
        random_labels = torch.full(
            (all_samples.shape[0],), class_i, dtype=torch.int64, device=network.device
        )
        fake_samples = generator((noise, random_labels))
        _, features  = discriminator(fake_samples)
        act2         = features.cpu().numpy()

        del(all_samples, noise, random_labels, fake_samples, features)


    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        print(f"[!] FID calculation produces a singular product; adding {1e-6} to diagonal of cov estimates")
        offset  = np.eye(sigma1.shape[0]) * 1e-6
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"[!] FID calculation produces complex value: {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    ###################################################################################################################

    print(f"\tFID score: {fid}")
