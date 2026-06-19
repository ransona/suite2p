# Ranson Lab Suite2p Installation

This repository contains the Ranson Lab Suite2p version. It includes local
registration, detection, and GUI changes that are not present in the upstream
PyPI package. The lab version is maintained on the `main` branch of
<https://github.com/ransona/suite2p>.

## Install with Conda

Install Miniforge or another Conda distribution first. Then open a terminal (or
Anaconda Prompt on Windows) and run:

```bash
conda create --name suite2p_lab python=3.11 -y
conda activate suite2p_lab
python -m pip install --upgrade pip
python -m pip install "suite2p[gui,io] @ git+https://github.com/ransona/suite2p.git@main"
```

This installs Suite2p and its GUI and input/output dependencies. Cloning the
repository is not required for normal use.

## Current Ubuntu Server GPU Installation

This section records the working Ranson Lab server configuration as of
2026-06-19. Update the versions and commands here when the server driver or
software stack changes.

The tested server configuration is:

- Ubuntu 24.04.4 LTS, x86_64
- Linux kernel 6.17.0-29-generic
- Two NVIDIA GeForce RTX 4090 GPUs (compute capability 8.9)
- NVIDIA driver 580.159.04
- CUDA 13.0 reported by `nvidia-smi`
- Python 3.11.15
- Torch 2.12.0 with CUDA 13.0 (`2.12.0+cu130`)
- torchvision 0.27.0
- Triton 3.7.0
- cuDNN 9.20.0

For a new environment on this server, use the following sequence instead of
allowing pip to choose its default Torch build:

```bash
conda create --name suite2p_lab python=3.11 -y
conda activate suite2p_lab
python -m pip install --upgrade pip

python -m pip install \
    torch==2.12.0 \
    torchvision==0.27.0 \
    --index-url https://download.pytorch.org/whl/cu130

python -m pip install "suite2p[gui,io] @ git+https://github.com/ransona/suite2p.git@main"
```

The PyTorch CUDA wheel installs its required CUDA 13 runtime libraries inside
the Conda environment. A separate system CUDA toolkit is not required for
normal Suite2p use, but the host NVIDIA driver must support this CUDA build.
The current server uses driver 580.159.04.

Confirm the host driver before installation:

```bash
nvidia-smi
```

After installation, verify the exact Torch build and execute an operation on
the GPU:

```bash
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("Torch CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))

assert torch.__version__ == "2.12.0+cu130"
assert torch.version.cuda == "13.0"
assert torch.cuda.is_available()
value = (torch.ones(1024, device="cuda") * 2).sum().item()
assert value == 2048
print("CUDA tensor test passed")
PY
```

Expected key output on the current server is:

```text
Torch: 2.12.0+cu130
Torch CUDA runtime: 13.0
CUDA available: True
GPU count: 2
CUDA tensor test passed
```

The NVIDIA runtime packages, Triton, and cuDNN are dependencies of the pinned
Torch wheel and should not normally be installed or pinned individually.

Start the GUI with:

```bash
conda activate suite2p_lab
suite2p
```

The environment must be activated each time a new terminal is opened.

## Verify the Installation

Run:

```bash
python -c "import suite2p; print(suite2p.__file__)"
python -m pip freeze | grep -i suite2p
```

On Windows, replace the second command with:

```bat
python -m pip freeze | findstr /I suite2p
```

The package record should contain `github.com/ransona/suite2p`, not only a PyPI
version number or `MouseLand/suite2p`.

## Check GPU Support

On a machine with an NVIDIA GPU, run:

```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

`CUDA available: True` confirms that Torch can use an NVIDIA GPU. If it reports
`False`, follow the PyTorch installation selector at
<https://pytorch.org/get-started/locally/> to install the appropriate CUDA build
inside the activated `suite2p_lab` environment. Suite2p can still run on CPU.

To select a particular GPU on Linux, launch Suite2p with, for example:

```bash
CUDA_VISIBLE_DEVICES=1 suite2p
```

## Upgrade the Lab Version

To update an existing environment to the current `main` branch:

```bash
conda activate suite2p_lab
python -m pip install --upgrade --force-reinstall --no-deps "suite2p @ git+https://github.com/ransona/suite2p.git@main"
```

Run the verification commands again after upgrading.

## Editable Installation for Developers

Only clone the repository when modifying Suite2p itself:

```bash
git clone https://github.com/ransona/suite2p.git
cd suite2p
conda create --name suite2p_lab python=3.11 -y
conda activate suite2p_lab
python -m pip install --upgrade pip
python -m pip install -e ".[gui,io]"
```

An editable installation imports the code directly from the cloned directory,
so local edits take effect without reinstalling the package.

## Remove and Reinstall

For a clean reinstall, remove the environment and repeat the installation:

```bash
conda deactivate
conda env remove --name suite2p_lab
conda create --name suite2p_lab python=3.11 -y
conda activate suite2p_lab
python -m pip install --upgrade pip
python -m pip install "suite2p[gui,io] @ git+https://github.com/ransona/suite2p.git@main"
```

Do not use `pip install suite2p` for the lab version. That command installs the
upstream PyPI release instead.
