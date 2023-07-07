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


"""Utilities for remote sensing datasets.

This module contains all neccessary classes and functions to load and use remote sensing datasets for classification
tasks.

Attributes
----------
AVAIL_DATASETS : dict
    A dictionary that specifies all available datasets. There is one entry per dataset, which is a dictionary that 
    contains the following information:

    - "format" : str
        The format of the dataset. Only "raw" images are supported at the moment.

    - "image" : list
        A list that contains the name of the file that contains the image data.

    - "gt" : list
        A list that contains the name of the file that contains the ground-truth data.
        
    - "segmentation" : list
        A list that contains the name of the file that contains the segmentation map.

    - "classes" : list
        A list that contains the names of the classes that are present in the dataset.
"""


__author__ = "alvrogd"


import math
import pickle
import random
import subprocess
import time

import numpy as np
import sklearn.decomposition
import sklearn.preprocessing
import torch
import torchvision.transforms.functional as TF

import cppimport


AVAIL_DATASETS = {
    # RAW image:
    #   --> All bands are stored in a single file
    #   --> The GT is a single file
    #
    "pavia_university": {
        "format": "raw",
        "image": [
            "dataset.raw",
        ],
        "gt": [
            "ground_truth.raw",
        ],
        "segmentation": [
            "segmentation.raw",
        ],
        "classes": [
            "asphalt",
            "meadows",
            "gravel",
            "trees",
            "painted_metal_sheets",
            "bare_soil",
            "bitumen",
            "self_blocking_bricks",
            "shadows",
        ],
    },
}


class HyperDataset(torch.utils.data.Dataset):
    """Remote sensing dataset for classification tasks.

    This class represents a remote sensing dataset that has been preprocessed and is ready to be used for
    classification tasks.

    Attributes
    ----------
    name : str
        The name of the dataset.

    name_no_slashes : str
        The name of the dataset in which slashes ("/") are replaced by underscores ("_").

    patch_size : int
        The size of the patches to be extracted from the dataset. It is measured in pixels along the patch's side.

    segmented : bool
        Whether the dataset has been segmented into superpixels or not.

    segmentation_map : numpy.ndarray
        The segmentation map of the dataset, if segmented into superpixels. It is a 2D array of integers, where the
        entry (i, j) represents the superpixel to which the pixel in row "i" and column "j" belongs.

    superpixels_count : int
        The number of superpixels in the dataset, if segmented into superpixels.

    superpixels_coordinates : dict
        The coordinates of the minimum rectangle that encloses each superpixel in the dataset, if segmented into
        superpixels.
        
        There are four keys in this dictionary that correspond the cardinal directions: "N", "W", "S" and "E". Inside
        each of these keys there is a list of integers, where the entry "i" represents the coordinate of the minimum
        rectangle that encloses the superpixel "i".
        
    image : numpy.ndarray
        The image data of the dataset. It is a 3D array of floats, where the entry (i, j, k) represents the value of
        the pixel in row "i", column "j" and band "k".

    gt : numpy.ndarray
        The ground truth data of the dataset. It is a 2D array of integers, where the entry (i, j) represents the
        class of the pixel in row "i" and column "j".

    height : int
        The height of the dataset. It is measured in pixels.

    width : int
        The width of the dataset. It is measured in pixels.

    bands : int
        The number of bands of the dataset.

    classes : list
        The list of classes of the dataset.

    classes_count : int
        The number of classes of the dataset.

    biggest_superpixel_size : int
        The size of the biggest superpixel in the dataset, if segmented into superpixels. It is measured in pixels
        contained by the minimum rectangle that encloses the superpixel.

    original_min : float
        The minimum value of the original image data.

    original_max : float
        The maximum value of the original image data.

    ratios : tuple
        Tuple of floats that represent the (1) training set ratio and (2) validation set ratio. They are specified in
        the [0.00, 1.00] range.

    train_set : dict
        Dictionary that contains the training set of the dataset. It has two keys: "samples" and "labels".
        
        The first key contains a list of samples, where each sample is a 3D numpy.ndarray of floats that represents
        the image data of a patch.
        
        The second key contains a list of labels, where each label is the class of the corresponding sample in the
        "samples" list.

    val_set : dict
        Dictionary that contains the validation set of the dataset. It has two keys: "samples" and "labels".
        
        The first key contains a list of samples, where each sample is a 3D numpy.ndarray of floats that represents
        the image data of a patch.
        
        The second key contains a list of labels, where each label is the class of the corresponding sample in the
        "samples" list.

    test_set : dict
        Dictionary that contains the test set of the dataset. It has two keys: "samples" and "labels".
        
        The first key contains a list of samples, where each sample is a 3D numpy.ndarray of floats that represents
        the image data of a patch.
        
        The second key contains a list of labels, where each label is the class of the corresponding sample in the
        "samples" list.

    train_count : int
        The number of samples in the training set.

    val_count : int
        The number of samples in the validation set.

    test_count : int
        The number of samples in the test set.

    current_set : dict
        Reference to the set of the dataset that is currently active. Each time a sample is requested, it is taken
        from this set.

    train_mode : bool
        True if the dataset is currently using the training set. False otherwise.

    pixel_level_labels : bool
        True if the dataset is retrieving pixel-level labels for requested samples. False otherwise.

    data_augmentation : bool
        True if the dataset is performing data augmentation on requested training samples. False otherwise.
    """

    def __init__(self, name, segmented=True, patch_size=32, ratios=(0.75, 0.05)):
        """Loads and preprocesses the requested remote sensing dataset.
        
        If requested, the dataset will be segmented into superpixels, and each superpixel containing labeled pixels
        will be used as a training/validation/test sample. Otherwise, each labeled pixel will be used as a sample.

        All image data is also scaled to the [-1, 1] range.

        By default, data augmentation is performed on the training set.

        All of the dataset's attributes are initialized in this method.

        Parameters
        ----------
        name : str
            The name of the dataset. It must be one of the datasets defined in "AVAIL_DATASETS".

        segmented : bool, optional
            Whether to segment the dataset into superpixels or not. Default is "True".

        patch_size : int, optional
            The size of the patches to be extracted from the dataset. It is measured in pixels along the patch's side.
            Default is "32".

        ratios : tuple, optional
            Tuple of floats that represent the (1) training set ratio and (2) validation set ratio. They are specified
            in the [0.00, 1.00] range. Default is "(0.75, 0.05)".
        """

        super(HyperDataset, self).__init__()


        self.name = name
        self.name_no_slashes = name.replace("/", "_") 

        self.patch_size = patch_size


        print(f"[*] Loading dataset {self.name} from disk")
        self.load_from_disk()

        print(f"[*] Recording available classes")
        self.record_classes()


        print(f"[*] Starting preprocessing")

        # At first, we assume that the dataset will not be segmented
        self.segmented = segmented
        self.segmentation_map = None
        self.superpixels_count = 0
        self.superpixels_coordinates = None

        if self.segmented:
            print(f"[*] Segmenting dataset into superpixels")
            self.segment()    

        print(f"[*] Scaling dataset to [-1, 1]")
        self.scale()   

        print(f"[*] Splitting dataset into train, validation, and test sets: ratios {ratios}")
        self.split_dataset(ratios)

        # By default, the training set is the one from which samples are returned
        self.to_train()
        # Data augmentation is applied to the training set
        self.set_data_augmentation(True)
        # And the labels of superpixel samples will be those decided by majority-voting
        self.set_pixel_level_labels(False)


    def load_from_disk(self):
        """Loads the specified dataset from disk.

        This function uses the "name" attribute to load the corresponding dataset from disk, storing in memory the
        image data and the ground truth data. It also records the height, width, and number of bands of the dataset.
        """

        self.image, self.gt = read_unprocessed_dataset(self.name)
        
        # The image is returned in band-format, and the GT does not have multiple bands
        assert self.image.shape[1 : 3] == self.gt.shape[: 2]

        self.height, self.width, self.bands = self.image.shape[1], self.image.shape[2], self.image.shape[0]


    def record_classes(self):
        """Records the available classes in the dataset as attributes."""

        self.classes       = AVAIL_DATASETS[self.name]["classes"]
        self.classes_count = len(self.classes)


    def segment(self):
        """Segments the dataset into superpixels.

        This function segments the dataset into superpixels, following the segmentation map listed in the
        "AVAIL_DATASETS" dictionary.
        
        The provided segmentation map is stored in memory, as well as the number of segments, and the coordinates of
        the minimum rectangle that encloses each one.        
        """

        # Contains auxiliary procedures written in C++ to speed up the preprocessing
        datasets_helper = cppimport.imp("datasets_helper")


        self.segmented = True


        print(f"\t[*] Reading segmentation map from disk")

        with open(f"datasets/{self.name}/{AVAIL_DATASETS[self.name]['segmentation'][0]}", "rb") as input:
            self.segmentation_map = input.read()

        
        # The segmentation map contains:
        #   - 2 ints at the beginning that specify the width and height of the image, respectively
        #   - 1 int per pixel, in row-major order, that tells to which superpixel the pixel belongs to
        #
        # We are not interested in the two first entries
        self.segmentation_map = np.frombuffer(self.segmentation_map, dtype=np.int32)
        self.segmentation_map = np.reshape(self.segmentation_map[2 :], (self.height, self.width))

        # All superpixels are identified by an integer number that starts in 0
        self.superpixels_count = np.max(self.segmentation_map) + 1


        # The dictionary will store the coordinates of the minimum rectangle that encloses each superpixel
        coordinates = datasets_helper.gather_superpixels_coordinates(self.segmentation_map, self.superpixels_count)

        self.superpixels_coordinates = {
            "N": np.frombuffer(coordinates, dtype=np.int32, count=self.superpixels_count,
                                offset=0).copy(),
            "W": np.frombuffer(coordinates, dtype=np.int32, count=self.superpixels_count,
                                offset=4 * self.superpixels_count * 1).copy(),
            "S": np.frombuffer(coordinates, dtype=np.int32, count=self.superpixels_count,
                                offset=4 * self.superpixels_count * 2).copy(),
            "E": np.frombuffer(coordinates, dtype=np.int32, count=self.superpixels_count,
                                offset=4 * self.superpixels_count * 3).copy(),
        }
        
        datasets_helper.free_superpixels_coordinates()

        self.biggest_superpixel_size = self.get_biggest_superpixel_size()


    def scale(self):
        """Scales the image data to the range [-1, 1]."""

        # All data gets scaled to the range [-1, 1]
        self.original_min = np.min(self.image)
        self.original_max = np.max(self.image)
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=False)
        
        # We need all values as a single-dimensional array
        self.image = np.reshape(self.image, (self.bands * self.height * self.width, 1))
        self.image = scaler.fit_transform(self.image)
        self.image = np.reshape(self.image, (self.bands, self.height, self.width))


    def get_biggest_superpixel_size(self):
        """Computes the size of the minimum rectangle that encloses the biggest superpixel.

        The size is measured as the number of pixels that the rectangle contains.

        Returns
        -------
        int
            The size of the biggest superpixel.
        """

        biggest_size = -1


        if self.segmented:

            for superpixel in range(self.superpixels_count):

                N = self.superpixels_coordinates["N"][superpixel]
                W = self.superpixels_coordinates["W"][superpixel]
                S = self.superpixels_coordinates["S"][superpixel]
                E = self.superpixels_coordinates["E"][superpixel]

                current_height = S - N + 1
                current_width  = E - W + 1

                current_size = current_height * current_width
                biggest_size = max(biggest_size, current_size)

        
        return biggest_size


    def split_dataset(self, ratios):
        """Splits the samples into training, validation and test sets.

        This function extracts all available labeled samples from the dataset, and splits them into training,
        validation and test sets. The samples are randomly selected.

        Parameters
        ----------
        ratios : tuple
            Tuple of floats that represent the (1) training set ratio and (2) validation set ratio. They are specified
            in the [0.00, 1.00] range.
        """

        # Contains auxiliary procedures written in C++ to speed up the preprocessing
        datasets_helper = cppimport.imp("datasets_helper")


        self.ratios = ratios
        train_ratio, val_ratio = self.ratios[0], self.ratios[1]


        # First of all, we need to gather all the available samples for each class, according to the GT
        #
        # This function returns a list that contains a sublist for each class
        # Each class' sublist contains enough info to identify and retrieve the corresponding samples
        all_samples = datasets_helper.gather_all_samples(
                          self.gt,
                          self.classes_count,
                          self.segmented,
                          self.segmentation_map if self.segmented else np.zeros((1, 1), dtype=np.int32),
                          self.superpixels_count,
                          self.superpixels_coordinates["N"] if self.segmented else np.zeros((1, 1), dtype=np.int32),
                          self.superpixels_coordinates["W"] if self.segmented else np.zeros((1, 1), dtype=np.int32),
                          self.superpixels_coordinates["S"] if self.segmented else np.zeros((1, 1), dtype=np.int32),
                          self.superpixels_coordinates["E"] if self.segmented else np.zeros((1, 1), dtype=np.int32)
                      )


        self.train_set = { "samples": [], "labels": [] }
        self.val_set   = { "samples": [], "labels": [] }
        self.test_set  = { "samples": [], "labels": [] }


        for class_i, samples in enumerate(all_samples):

            print(f"\t[*] Recording samples for class {self.classes[class_i]} ({len(samples)} items)")

            random.shuffle(samples)

            train = samples[0                                             : int(len(samples) * train_ratio)]
            val   = samples[int(len(samples) * train_ratio)               : int(len(samples) * (train_ratio + val_ratio))]
            test  = samples[int(len(samples) * (train_ratio + val_ratio)) :]

            self.train_set["samples"] += train
            self.val_set["samples"]   += val
            self.test_set["samples"]  += test

            self.train_set["labels"] += [class_i] * len(train)
            self.val_set["labels"]   += [class_i] * len(val)
            self.test_set["labels"]  += [class_i] * len(test)

        print("")


        self.train_set["samples"] = np.array(self.train_set["samples"], dtype=np.int32)
        self.train_set["labels"]  = np.array(self.train_set["labels"], dtype=np.int32)

        self.val_set["samples"] = np.array(self.val_set["samples"], dtype=np.int32)
        self.val_set["labels"]  = np.array(self.val_set["labels"], dtype=np.int32)

        self.test_set["samples"] = np.array(self.test_set["samples"], dtype=np.int32)
        self.test_set["labels"]  = np.array(self.test_set["labels"], dtype=np.int32)


        self.train_count = len(self.train_set["samples"])
        self.val_count   = len(self.val_set["samples"])
        self.test_count  = len(self.test_set["samples"])


    def to_train(self):
        """Sets the current dataset to the training set."""

        self.current_set = self.train_set
        self.train_mode  = True


    def to_validation(self):
        """Sets the current dataset to the validation set."""

        self.current_set = self.val_set
        self.train_mode  = False


    def to_test(self):
        """Sets the current dataset to the test set."""

        self.current_set = self.test_set
        self.train_mode  = False


    def set_pixel_level_labels(self, new_state):
        """Updates the value of the "pixel_level_labels" attribute."""

        self.pixel_level_labels = new_state

    
    def set_data_augmentation(self, new_state):
        """Updates the value of the "data_augmentation" attribute."""

        self.data_augmentation = new_state


    def __len__(self):
        """Returns the number of samples in the current set.

        Returns
        -------
        int
            Number of samples in the current set.
        """
        return len(self.current_set["samples"])


    def __getitem__(self, item):
        """Returns the sample and its label at the specified index

        The requested sample is taken from the current active set.

        Parameters
        ----------
        item : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            Tuple that contains (1) the sample, (2) its label, and (3) an array of pixel-level labels if activated and
            the sample is a superpixel. If not activated, the third element is an empty array.
        """

        sample, label, labels_pixel_level = None, None, np.zeros(1, dtype=np.int64)


        # The class to which the sample belongs is stored in the "labels" array
        # --> warning: PyTorch expects torch.long/torch.int64 datatype on labels
        label = self.current_set["labels"][item]
        label = np.array(label, dtype=np.int64)


        # If the sample is a pixel...
        if not self.segmented:

            # The "samples" array contains the coordinates of the requested pixel
            row, col = self.current_set["samples"][item][0], self.current_set["samples"][item][1]

            # The resulting patch is going to be centered on the requested pixel
            # We need to handle spacial cases in which the patch extends beyond the image boundaries
            n_row            = row - self.patch_size // 2
            n_padding, n_row = -n_row if n_row < 0 else 0, max(0, n_row)
            
            s_row            = row + int(math.ceil(self.patch_size / 2))
            s_padding, s_row = s_row - self.height if s_row > self.height else 0, min(self.height, s_row)
            
            w_col            = col - self.patch_size // 2
            w_padding, w_col = -w_col if w_col < 0 else 0, max(0, w_col)
            
            e_col            = col + int(math.ceil(self.patch_size / 2))
            e_padding, e_col = e_col - self.width if e_col > self.width else 0, min(self.width, e_col)

            sample = np.array(self.image[:, n_row : s_row, w_col : e_col], dtype=np.float32)
            sample = np.pad(sample, ((0, 0), (n_padding, s_padding), (w_padding, e_padding)), mode="edge")

        # Otherwise, the sample will correspond to a superpixel
        else:

            # The "samples" array contains the ID of the requested superpixel
            superpixel = self.current_set["samples"][item][0]

            # There will be one patch per superpixel
            # First, we need to compute the coordinates of the central pixel of the superpixel, as the patch is going
            # to be centered on it
            N = self.superpixels_coordinates["N"][superpixel]
            W = self.superpixels_coordinates["W"][superpixel]
            S = self.superpixels_coordinates["S"][superpixel]
            E = self.superpixels_coordinates["E"][superpixel]

            row, col = N + (S - N) // 2, W + (E - W) // 2

            # We need to handle spacial cases in which the patch extends beyond the image boundaries
            n_row            = row - self.patch_size // 2
            n_padding, n_row = -n_row if n_row < 0 else 0, max(0, n_row)
            
            s_row            = row + int(math.ceil(self.patch_size / 2))
            s_padding, s_row = s_row - self.height if s_row > self.height else 0, min(self.height, s_row)
            
            w_col            = col - self.patch_size // 2
            w_padding, w_col = -w_col if w_col < 0 else 0, max(0, w_col)
            
            e_col            = col + int(math.ceil(self.patch_size / 2))
            e_padding, e_col = e_col - self.width if e_col > self.width else 0, min(self.width, e_col)

            sample = np.array(self.image[:, n_row : s_row, w_col : e_col], dtype=np.float32)
            sample = np.pad(sample, ((0, 0), (n_padding, s_padding), (w_padding, e_padding)), mode="edge")


            # One more thing...
            #
            # If pixel_level_labels==True, we are also going to return an array that contains the labels of all the
            # pixels that belong to the superpixel, in addition to the label that was decided via majority voting
            #
            #   --> this allows computing pixel-level accuracies on superpixels
            if self.pixel_level_labels == True:

                # All arrays for all possible samples need to have the same dimensions; otherwise, the DataLoader will
                # crash when stacking multiple samples to generate a single batch
                #
                #   --> (-1 is a sentinel value to point out that there are no more labels in the array)
                labels_pixel_level = np.full(self.biggest_superpixel_size + 1, -1, dtype=np.int64)

                counter = 0

                for row in range(N, S + 1):
                    for col in range(W, E + 1):

                        # If the current pixel belongs to the superpixel
                        if self.segmentation_map[row, col] == superpixel:

    ############################ For compatibility with https://doi.org/10.3390/rs13142687 ############################
                            # If we are retrieving samples from the train set, we need to exclude the central pixel
                            if self.train_mode == True and row == N + (S - N) // 2 and col == W + (E - W) // 2:
                                continue
    ###################################################################################################################

                            # All labels are subtracted '1' from them when generating the samples
                            labels_pixel_level[counter] = self.gt[row, col] - 1
                            counter += 1


        # Conversion to PyTorch tensors
        sample             = torch.from_numpy(sample)
        label              = torch.from_numpy(label)
        labels_pixel_level = torch.from_numpy(labels_pixel_level)


        # Data augmentation is only performed to increase the complexity of the training data
        # --> we do not want to hinder the testing phases
        if self.train_mode and self.data_augmentation:
            # The function works on PyTorch tensors
            sample = apply_augmentation(sample)


        return sample, label, labels_pixel_level


    def __str__(self):
        return "[*] HyperDataset summary:\n" \
               f"\tName: {self.name}\n" \
               f"\tShape: (height) {self.height}, (width) {self.width}, (bands) {self.bands}\n" \
               f"\tClasses: {self.classes}\n" \
               f"\tClasses count: {self.classes_count}\n" \
               f"\tSegmented: {self.segmented}\n" \
               f"\tSuperpixels count: {self.superpixels_count}\n" \
               f"\tPatch size: {self.patch_size}\n" \
               f"\tRatios: (train) {self.ratios[0]}, (val) {self.ratios[1]}\n" \
               f"\tSamples count: (train) {self.train_count}, (val) {self.val_count}, (test) {self.test_count}\n"


    def write_on_disk(self):
        """Stores the dataset on disk.

        The dataset stores itself on disk as a pickle file, recording all of its internal state.
        
        This function is intended to avoid preprocessing a certain dataset every time that is needed for a
        classification pipeline.

        The filename is "preprocessed/hyperdataset".
        """

        with open(f"preprocessed/hyperdataset", "wb") as output:

            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


def apply_augmentation(sample):
        """Applies data augmentation techniques to a sample.

        Several different transformation techniques are randomly applied to the given sample to generate a synthetic
        one. The possible transformations are:

        - A rotation of 0, 90, 180 or 270 degrees.
        - A horizontal flip, with a 50% probability.
        - A vertical flip, with a 50% probability.

        Parameters
        ----------
        sample : torch.Tensor
            The sample to be augmented.

        Returns
        -------
        torch.Tensor
            The augmented sample.
        """

        # Rotation
        angle = random.choice([90, 180, 270, 360])

        if angle != 360:
            sample = TF.rotate(sample, angle)

        # Horizontal flip
        if random.random() > 0.5:
            sample = TF.hflip(sample)

        # Vertical flip
        if random.random() > 0.5:
            sample = TF.vflip(sample)

        return sample


def read_unprocessed_dataset(name):
    """Loads the specified dataset from disk.

    Parameters
    ----------
    name : str
        The name of the dataset. It must be one of the datasets defined in "AVAIL_DATASETS".

    Returns
    -------
    tuple
        A tuple containing the following elements:
        
        1. A numpy.ndarray containing the image data.
        2. A numpy.ndarray containing the ground truth data.
    """

    image, gt = None, None


    if AVAIL_DATASETS[name]["format"] == "raw":

        print(f"\t{name} dataset is in RAW format")


        # The image is stored using a custom format
        with open(f"datasets/{name}/{AVAIL_DATASETS[name]['image'][0]}", "rb") as input:

            # The first three 4-byte entries are the band count, width and height of the file respectively
            data = np.fromfile(input, dtype=np.int32)
            bands, width, height = data[0], data[1], data[2]

            # After that, there is one 4-byte entry per band in each pixel-vector
            #   (the pixel-vectors are placed one after another)
            image = np.reshape(data[3 :], (height, width, bands)).astype(np.float32)

            # The image must be converted to band-format
            image = np.transpose(image, (2, 0, 1))


        # The GT is stored using a custom format
        with open(f"datasets/{name}/{AVAIL_DATASETS[name]['gt'][0]}", "rb") as input:

            # The first three 4-byte entries are the width, height and band count respectively
            # After that, there is one 4-byte entry per pixel, that tells which class it belongs to
            gt = np.fromfile(input, dtype=np.int32, offset=12)
            gt = np.reshape(gt, (height, width))


    else:

        raise Exception("Unsupported dataset format")


    return (image, gt)


def read_preprocessed_dataset(path):
    """Reads a preprocessed HyperDataset from disk.
    
    This function loads a preprocessed HyperDataset from disk that has been stored as a pickle file.

    Parameters
    ----------
    path : str
        Relative path to the file.

    Returns
    -------
    HyperDataset
        The preprocessed HyperDataset.
    """

    dataset = None

    with open(path, "rb") as input:
        dataset = pickle.load(input)

    return dataset
