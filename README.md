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



---

## How To

#

### Add files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.lrz.de/smlr_ws22-23/g2-peg-in-hole.git
git branch -M main
git push -uf origin main
```
#

### Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.lrz.de/smlr_ws22-23/g2-peg-in-hole/-/settings/integrations)

#

### Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

#

### Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

---

## To be filled out

#

### Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

#

### Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

#

### Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

#

### Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

#

### Authors and acknowledgment
Show your appreciation to those who have contributed to the project.