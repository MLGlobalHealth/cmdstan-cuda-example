
# cmdstan-cuda-example


## About this repository

A simple example of offloading computations in a Stan model to the GPU. Uses the beta negative binomial distribution and its gradient as an example.


## Preliminaries

This repository is written for CmdStan version 2.35.0. For python users, here's install script. Installation scripts via other interfaces are similar to this one.

```python
from cmdstanpy import cmdstan_path, set_cmdstan_path, install_cmdstan
install_cmdstan(version="2.35.0", verbose=True, progress=True, cores=4)
set_cmdstan_path(os.path.join(os.path.expanduser('~'), '.cmdstan', 'cmdstan-2.35.0'))
cmdstan_path()
```

Ensure that NVIDIA GPU drivers are correctly installed.

```sh
nvidia-smi
```

Ensure that CUDA Toolkit are correctly installed and is accessible via PATH

```sh
nvcc --version
echo $CUDA_HOME
```


## Quick start

Clone this repository, rename it to `cmdstan-cuda`, and move it to the root directory of CmdStan.
In most cases, this will be located at: `$HOME/.cmdstan/cmdstan-2.35.0`.

```sh
git clone https://github.com/MLGlobalHealth/cmdstan-cuda-example.git cmdstan-cuda
CMDSTAN_HOME="$HOME/.cmdstan/cmdstan-2.35.0"
mv cmdstan-cuda "$CMDSTAN_HOME/"
```

Execute `main.py`, and go to the `output` and `model` folders to have a look.

The `output` folder should contain a comparison histogram.

The `model` folder should contain the saved results, logs, and diagnostics.

## Profiling

To find the critical point on your current machine where CPU and GPU performance are equal, run:

```sh
python prof/prof.py
```

Then go to the `output` folder to check. It should contain CSV output of the runtime test and comparative plots.

To view the details and timeline of CUDA API calls by CmdStan model, run:

```sh
nsys profile --stats=true --force-overwrite=true -o report ./prof_cu_bnb.sh
```

Could use NVIDIA Nsight Systems provided to open the generated `report.nsys-rep` file.






