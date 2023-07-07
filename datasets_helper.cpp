// Copyright 2023 Álvaro Goldar Dieste

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


/**
 * @file   datasets_helper.cpp
 * @author alvrogd
 * @brief  This unit implements various auxiliary functions written in C++ that are used by
 *         datasets.py, in order to speed-up the preprocessing stage of datasets.
 */


// To export this unit as a Python module
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <limits>


namespace py = pybind11;


int32_t *superpixels_coordinates = nullptr;


/**
 * @brief Gathers the coordinates of the minimum rectangles that enclose each superpixel.
 * 
 * @param seg_map_as_nparr  A NumPy array that represents the segmentation map.
 * @param superpixels_count How many superpixels there are in the segmentation map.
 *
 * @return A memory view of the computed coordinates.
 */
py::memoryview gather_superpixels_coordinates(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> seg_map_as_nparr,
    int32_t superpixels_count
)
{
    py::buffer_info seg_map_np = seg_map_as_nparr.request();

    int32_t *seg_map_data = reinterpret_cast<int32_t *>(seg_map_np.ptr);
    int32_t image_height  = static_cast<int32_t>(seg_map_np.shape[0]);
    int32_t image_width   = static_cast<int32_t>(seg_map_np.shape[1]);


    int32_t *coordinates = new int32_t[superpixels_count * 4];
    int32_t *N = coordinates;
    int32_t *W = coordinates + superpixels_count;
    int32_t *S = coordinates + superpixels_count * 2;
    int32_t *E = coordinates + superpixels_count * 3;

    std::fill(N, N + superpixels_count, std::numeric_limits<int32_t>::max());
    std::fill(W, W + superpixels_count, std::numeric_limits<int32_t>::max());
    std::fill(S, S + superpixels_count, std::numeric_limits<int32_t>::min());
    std::fill(E, E + superpixels_count, std::numeric_limits<int32_t>::min());


    for(int32_t row = 0; row < image_height; ++row) {
        for(int32_t col = 0; col < image_width; ++col) {

            // Superpixel ID
            int32_t superpixel = seg_map_data[row * image_width + col];

            // Update corresponding coordinates if necessary
            if(row < N[superpixel]) {
                N[superpixel] = row;
            }

            if(col < W[superpixel]) {
                W[superpixel] = col;
            }

            if(row > S[superpixel]) {
                S[superpixel] = row;
            }

            if(col > E[superpixel]) {
                E[superpixel] = col;
            }
        }
    }


    // To be able to free the allocated memory later
    superpixels_coordinates = coordinates;

    return py::memoryview::from_memory(
        coordinates,
        (superpixels_count * 4) * sizeof(int32_t)
    );
}


/**
 * @brief Frees the memory allocated by gather_superpixels_coordinates().
 */
void free_superpixels_coordinates()
{
    delete[] superpixels_coordinates;
}


/**
 * @brief Scans the whole dataset to gather as much labeled samples as possible.
 *
 * @details
 *  If the image has not been segmented, each labeled pixel is treated as a sample. Otherwise,
 *  superpixels that contain at least one labeled pixel are considered as valid samples. A given
 *  superpixel's class is determined through a majority voting using all of the superpixel's
 *  labeled pixels.
 * 
 * @param gt_as_nparr               A NumPy array that represents the ground-truth.
 * @param classes_count             How many different classes can be found in the dataset.
 * @param segmented                 If the image has been segmented or not.
 * @param seg_map_as_nparr          A NumPy array that represents the segmentation map.
 * @param superpixels_count         How many superpixels there are in the segmentation map.
 * @param superpixels_coordinates_N A NumPy Array that represents the north coordinates of the
 *                                  minimal rectangle that encloses each superpixel.
 * @param superpixels_coordinates_W A NumPy Array that represents the west coordinates of the
 *                                  minimal rectangle that encloses each superpixel.
 * @param superpixels_coordinates_S A NumPy Array that represents the south coordinates of the
 *                                  minimal rectangle that encloses each superpixel.
 * @param superpixels_coordinates_E A NumPy Array that represents the east coordinates of the
 *                                  minimal rectangle that encloses each superpixel.
 *
 * @returns A list that contains as many lists as possible classes there are. Each class' list
 *          contains all of its identified samples.
 *
 *          If the image has not been segmented, a sample consists in the [row, col] coordinates
 *          of the labeled pixel. If the image has been segmented, a sample corresponds to the
 *          [integer] by which the segment can be identified.
 */
std::vector<std::list<std::vector<int>>> gather_all_samples(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> gt_as_nparr,
    int32_t classes_count,
    bool segmented,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> seg_map_as_nparr,
    int32_t superpixels_count,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> superpixels_coordinates_N,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> superpixels_coordinates_W,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> superpixels_coordinates_S,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> superpixels_coordinates_E
)
{
    py::buffer_info gt_np = gt_as_nparr.request();

    int32_t *gt_data     = reinterpret_cast<int32_t *>(gt_np.ptr);
    int32_t image_height = static_cast<int32_t>(gt_np.shape[0]);
    int32_t image_width  = static_cast<int32_t>(gt_np.shape[1]);

    int32_t *seg_map_data = reinterpret_cast<int32_t *>(seg_map_as_nparr.request().ptr);

    int32_t *N_data = reinterpret_cast<int32_t *>(superpixels_coordinates_N.request().ptr);
    int32_t *W_data = reinterpret_cast<int32_t *>(superpixels_coordinates_W.request().ptr);
    int32_t *S_data = reinterpret_cast<int32_t *>(superpixels_coordinates_S.request().ptr);
    int32_t *E_data = reinterpret_cast<int32_t *>(superpixels_coordinates_E.request().ptr);


    // One entry per class
    std::vector<std::list<std::vector<int>>> all_samples(classes_count,
                                                         std::list<std::vector<int>>()
                                                        );
    int32_t total_samples = 0;


    if(!segmented) {

        for(int32_t row = 0; row < image_height; ++row) {
            for(int32_t col = 0; col < image_width; ++col) {

                int32_t current_label = gt_data[row * image_width + col];

                // 0 means that the current pixel has not been labeled
                if(current_label > 0) {

                    std::vector<int32_t> sample({row, col});
                    all_samples[current_label - 1].push_back(sample);

                    ++total_samples;
                }
            }
        }
    }

    else {

        // Keep in mind that a majority voting will be performed to determine the class of each
        // superpixel
        std::vector<int> classes_occurrences(classes_count, 0);
        
        for(int32_t superpixel = 0; superpixel < superpixels_count; ++superpixel) {

            std::fill(classes_occurrences.begin(), classes_occurrences.end(), 0);

            // We will scan the rectangular area that encloses the superpixel
            for(int32_t row = N_data[superpixel]; row < S_data[superpixel] + 1; ++row) {

                for(int32_t col = W_data[superpixel]; col < E_data[superpixel] + 1; ++col) {

                    // If the scanned pixel belongs to the current superpixel
                    //   (superpixels are not perfect squares)
                    if(seg_map_data[row * image_width + col] == superpixel) {

                        // The pixel will participate in the majority voting if it has been
                        // labeled
                        if(gt_data[row * image_width + col] > 0) {
                            classes_occurrences[gt_data[row * image_width + col] - 1] += 1;
                        }
                    }
                }
            }

            // The superpixel needs to contain at least one labeled pixel in order to determine
            // its class (otherwise, the majority voting does not make sense)
            if(std::accumulate(classes_occurrences.begin(), classes_occurrences.end(), 0) > 0)
            {
                // If multiple classes end in a tie, we will just take the first one
                int32_t most_voted_class = std::distance(classes_occurrences.begin(),
                                                        std::max_element(
                                                            classes_occurrences.begin(),
                                                            classes_occurrences.end()
                                                        )
                                           );

                std::vector<int32_t> sample({superpixel});
                all_samples[most_voted_class].push_back(sample);

                ++total_samples;
            }
        }
    }


    py::print("[*] Total samples:", total_samples);


    return all_samples;
}


PYBIND11_MODULE(datasets_helper, m) {
    m.def("gather_superpixels_coordinates", &gather_superpixels_coordinates);
    m.def("free_superpixels_coordinates", &free_superpixels_coordinates);
    m.def("gather_all_samples", &gather_all_samples);
}


/*
<%
setup_pybind11(cfg)
cfg['extra_compile_args'] = ['-std=c++2a', '-O3', '-Wall', '-Wextra']
%>
*/
