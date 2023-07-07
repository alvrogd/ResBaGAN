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


FROM nvcr.io/nvidia/pytorch:22.04-py3

ENV DEBIAN_FRONTEND=noninteractive
# Prevents Python from writing bytecode files during execution
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/ResBaGAN/

RUN apt-get update
RUN apt-get install -y tmux
RUN pip install pip==22.1.2

# To report info about neural models
RUN pip install torch-summary==1.4.5

# To call datasets_helper.cpp from Python
RUN pip install pybind11==2.9.2
RUN pip install cppimport==22.5.11

# To automate, track and compare experiments
RUN pip install guildai==0.8.2
RUN guild completion --install --shell bash
ENV GUILD_HOME "/opt/ResBaGAN/.guildai"
