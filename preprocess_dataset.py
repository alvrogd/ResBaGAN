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


"""Reads and preprocesses a given dataset for classification.

More specifically, this script:

- Reads the specified dataset from disk.
- If requested, segments the dataset into superpixels.
- Scales the dataset to the [-1, 1] range.
- Splits the dataset into training, validation and test sets.
- Saves the preprocessed dataset to disk.
"""


__author__ = "alvrogd"


import argparse
import os
import time

import datasets


parser = argparse.ArgumentParser(
    prog="preprocess_dataset.py",
    description="Reads and preprocesses a given dataset for classification"
)

parser.add_argument(
    "--dataset",
    type=str,
    action="store",
    default="pavia_university",
)

parser.add_argument(
    "--segment",
    type=int,
    action="store",
    # 0: no segmentation, > 0: superpixel-based segmentation
    default=1
)

parser.add_argument(
    "--train_ratio",
    type=float,
    action="store",
    default=0.15
)

parser.add_argument(
    "--val_ratio",
    type=float,
    action="store",
    default=0.05
)

args = parser.parse_args()
print(f"[*] Arguments: {vars(args)}")


if not os.path.exists("preprocessed"):
    os.mkdir("preprocessed")


# Loading the requested dataset from disk and preprocessing it
t_preprocess_start = time.perf_counter()

dataset = datasets.HyperDataset(
    args.dataset, segmented=args.segment > 0, patch_size=32, ratios=(args.train_ratio, args.val_ratio)
)

t_preprocess_stop = time.perf_counter()
print(f"[time] Loading and preprocessing: {t_preprocess_stop - t_preprocess_start} s")

print(dataset)


# Lastly, to avoid preprocessing the same image each time that it is needed for a classification pipeline
print(f"[*] Storing the preprocessed dataset on disk")
dataset.write_on_disk()


print(f"[*] Done!")
