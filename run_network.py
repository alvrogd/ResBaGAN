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


"""Trains a certain network on a given dataset and evaluates its performance.

More specifically, this script:

- Trains the specified network on the given dataset.
- Evaluates the network in terms of classification accuracy, using OA, AA and Kappa metrics.

The trained model is also saved to disk.
"""


__author__ = "alvrogd"


import argparse
import os
import time

import torch

import datasets
import networks


parser = argparse.ArgumentParser(
    prog="run_network.py",
    description="Trains a certain network on a given dataset, and evaluates its performance"
)

parser.add_argument(
    "--dataset_path",
    type=str,
    action="store",
    # As saved by preprocess_dataset.py
    default="preprocessed/hyperdataset"
)

parser.add_argument(
    "--data_augmentation",
    type=int,
    action="store",
    # 0: no data augmentation, > 0: data augmentation
    default=1
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

args = parser.parse_args()
print(f"[*] Arguments: {vars(args)}")


if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists("preprocessed"):
    os.mkdir("preprocessed")


# Some arguments control the training and evaluation procedures
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


# Loading from disk the preprocessed dataset that will be used to train and test the network
dataset = datasets.read_preprocessed_dataset(args.dataset_path)
dataset.set_data_augmentation(args.data_augmentation > 0)
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


# Building the requested network
if args.network == "CNN2D":
    network = networks.CNN2D(dataset, hyperparams["device"], hyperparams)
elif args.network == "CNN2D_Residual":
    network = networks.CNN2D_Residual(dataset, hyperparams["device"], hyperparams)
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


# Training the network

t_train_start = time.perf_counter()

if args.network == "BAGAN" or args.network == "ResBaGAN":
    networks.dual_train_BAGAN(network, data_loader, dataset, hyperparams)
elif args.network == "ACGAN" or args.network == "ResACGAN":
    networks.dual_train_ACGAN(network, data_loader, dataset, hyperparams)
else:
    networks.train(network, data_loader, dataset, hyperparams)

t_train_stop = time.perf_counter()
print(f"[time] Training: {t_train_stop - t_train_start} s")


# Evaluating the network

if not dataset.segmented:

    print("[*] Computing pixel-level accuracies...")

    t_test_start = time.perf_counter()

    if args.network == "BAGAN" or args.network == "ResBaGAN" or args.network == "ACGAN" or args.network == "ResACGAN":
        networks.test(network.discriminator, data_loader, dataset, hyperparams, full_dataset=True)
    else:
        networks.test(network, data_loader, dataset, hyperparams, full_dataset=True)

    t_test_stop = time.perf_counter()
    print(f"[time] Testing (pixel-level accuracies): {t_test_stop - t_test_start} s")

else :

    print("[*] Computing segment-level accuracies...")

    t_test_start = time.perf_counter()

    if args.network == "BAGAN" or args.network == "ResBaGAN" or args.network == "ACGAN" or args.network == "ResACGAN":
        networks.test(network.discriminator, data_loader, dataset, hyperparams, full_dataset=True)
    else:
        networks.test(network, data_loader, dataset, hyperparams, full_dataset=True)

    t_test_stop = time.perf_counter()
    print(f"[time] Testing (segment-level accuracies): {t_test_stop - t_test_start} s")


    print("[*] Computing pixel-level accuracies...")

    t_test_start = time.perf_counter()

    if args.network == "BAGAN" or args.network == "ResBaGAN" or args.network == "ACGAN" or args.network == "ResACGAN":
        networks.test(network.discriminator, data_loader, dataset, hyperparams, multiple_labels=True, full_dataset=True)
    else:
        networks.test(network, data_loader, dataset, hyperparams, multiple_labels=True, full_dataset=True)

    t_test_stop = time.perf_counter()
    print(f"[time] Testing (pixel-level accuracies): {t_test_stop - t_test_start} s")


# Saving the trained network

torch.save(network.state_dict(), f"logs/{args.network}_model.pt")
