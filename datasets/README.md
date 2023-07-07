<!--
Copyright 2023 Álvaro Goldar Dieste

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


Modified "Pavia University" dataset, including ground truth and segmentation,
available for download at: https://nextcloud.citius.usc.es/s/CfEWMKfTfQAFmzB

The original "Pavia University" dataset was provided by Prof. Paolo Gamba from
the Telecommunications and Remote Sensing Laboratory at Pavia University,
Italy. It is hosted in the Hyperspectral Remote Sensing Scenes repository,
curated by M Graña, MA Veganzons, B Ayerdi, available at:
https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University

The files for the modified dataset are provided in a custom RAW format. Here
is an explanation of the contents of each file:

- `dataset.raw`: Contains the multispectral data. The beginning of the file
  features three 4-byte integers, representing the band count, the image
  width, and the image height, respectively. Next, the file contains one
  4-byte entry per band in each pixel, placing the bands of each pixel
  contiguously in a row-major order.

- `ground_truth.raw`: Contains the ground truth. At the start of the file,
  there are three 4-byte integers which represent the image width, image
  height, and band count (1). Following these, there is a 4-byte entry per
  pixel in a row-major order, indicating the class of the pixel. The classes
  are numbered from `1` to `num_classes`, with `0` meaning "no class".

- `segmentation.raw`: Contains the precomputed superpixel segmentation. The
  file starts with two 4-byte integers, specifying the image width and height.
  Then, the file contains a 4-byte entry per pixel in a row-major order,
  indicating the superpixel of the pixel. The superpixels are numbered from
  `0` to `num_segments - 1`.

Please refer to the following functions and their documentation for examples
on how to read these files:

- `datasets.py/read_unprocessed_dataset()`
- `datasets.py/HyperDataset.segment()`
