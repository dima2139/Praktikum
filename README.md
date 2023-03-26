# g2-peg-in-hole

## Installation

#

### Conda

* Download Anaconda from https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
* `bash Anaconda-latest-Linux-x86_64.sh`
```
conda create -n smlr python=3.9
conda activate smlr
```
* Context: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html

#

### Mujoco
* Download Mujoco from https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
```
tar -xf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
```
* Move folder `mujoco210/` into `~/.mujoco/`

```
pip install -U 'mujoco-py<2.2,>=2.1'
python
import mujoco_py
```
* If error `GLIBCXX_3.4.29 not found`:
```
rm ~/anaconda3/envs/smlr/lib/libstdc++.so.6
```

* If error `command 'gcc' failed with exit status 1`:
```
conda install -c conda-forge gcc
```

* Context: https://github.com/openai/mujoco-py/

#


### Robosuite
```
pip install robosuite
python -m robosuite.demos.demo_random_action
```
* If error `No such file or directory`:
```
cd ~
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
pip install -r requirements.txt
python -m robosuite.demos.demo_random_action
```
* If error `No such file or directory`:
```
python ~/anaconda3/envs/smlr/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py 
```

* Context: https://robosuite.ai/docs/installation.html


#

### PyTorch

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
* Context: https://pytorch.org/get-started/locally/


#

### StableBaselines3

```
pip install stable-baselines3[extra]
```

* Context: https://stable-baselines3.readthedocs.io/en/master/guide/install.html


#

### RAPS

* Dependencies:
```
sudo apt-get update
sudo apt-get install curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev
sudo apt-get install libglfw3-dev libgles2-mesa-dev patchelf
sudo mkdir /usr/lib/nvidia-000
```

* Add to `~/.bashrc`:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin
export MUJOCO_GL='egl'
export MKL_THREADING_LAYER=GNU
export D4RL_SUPPRESS_IMPORT_ERROR='1'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
```

* Environment:
```
conda create -n smlr3.7 python=3.7
conda activate smlr3.7
```

* Download DIFFERENT Mujoco from https://www.roboti.us/download/mujoco200_linux.zip
```
unzip mujoco200_linux.zip
mkdir ~/.mujoco
```

* Move folder `mujoco200/` into `~/.mujoco/`

* Download Mujoco activation key from https://www.roboti.us/file/mjkey.txt

* Move folder `mjkey.txt` into `~/.mujoco/`

* Install raps:
```
cd ~
git clone https://github.com/mihdalal/raps.git
cd raps
./setup_python_env.sh ~/raps
pip uninstall mujoco
pip install mujoco==2.1.5
```

* Context: https://www.roboti.us/index.html
* Context: https://github.com/mihdalal/raps



#

### Run End-to-End

```
python scripts/rl/sac/sacEval.py
```

(Corresponding video under videos/end-to-end)

#

### Run Action Primitives

```
python scripts/rap/sac/sacRapsEval.py
```

(Corresponding video under videos/primitives)
