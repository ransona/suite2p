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
