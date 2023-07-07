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


import torch


def check_cuda_availability():
    """Check if CUDA is available to PyTorch and return the current device if it is."""
    
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available")

    num_cuda_devices = torch.cuda.device_count()
    if num_cuda_devices == 0:
        raise Exception("No CUDA devices found. Please verify that your machine has a CUDA-capable GPU")

    current_device = torch.cuda.current_device()
    if current_device < 0:
        raise Exception(f"No CUDA device has been assigned to PyTorch. Available devices: {num_cuda_devices}")

    return current_device


if __name__ == "__main__":
    
    current_device = check_cuda_availability()

    print(f"Found {torch.cuda.device_count()} CUDA devices")
    print(f"CUDA device {torch.cuda.get_device_name(current_device)} [ID {current_device}, "
          f"{torch.cuda.device(current_device)}] has been assigned to PyTorch")
