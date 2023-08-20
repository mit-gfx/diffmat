<div align="center">
  <img width="500px" src="misc/diffmat_logo.png">
</div>
<br/>

[![Latest Release](https://img.shields.io/badge/diffmat-0.2.0-blue)]()

# Overview
***DiffMat*** is a [PyTorch](https://pytorch.org)-based differentiable procedural material modeling library that reproduces the compositing graph system in [Adobe Substance 3D Designer](https://www.adobe.com/products/substance3d-designer.html) with auto-differentiation. DiffMat automatically converts procedural materials in Substance's format (`*.sbs`) into differentiable computation graphs and optimizes individual node parameters to match user-captured material appearances (e.g., cellphone flash photos).

# Requirements

### Conda users

We provide an environment configuration file for [Anaconda](https://www.anaconda.com/products/distribution)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html) users. Clone the DiffMat repository and enter the source folder:
```bash
git clone git@github.com:mit-gfx/diffmat
cd diffmat
```
Create the virtual environment with:
```bash
conda env create -f environment.yml
conda activate diffmat
```

### Non-conda users

Create a virtual environment (e.g., using [venv](https://docs.python.org/3/library/venv.html)) or configure your existing environment to meet the following requirements.

* Python ≥ 3.7
* PyTorch ≥ 1.11.0
* Torchvision ≥ 0.12.0
* [Taichi-lang](https://www.taichi-lang.org) ≥ 1.3.0
* NumPy, SciPy, pandas, imageio, pyyaml, setuptools

Aside from PyTorch and Torchvision, all other packages are available via `pip install`:
```bash
pip install taichi numpy scipy pandas imageio pyyaml setuptools
```

### Optional packages

The following packages are *optional* unless you want to experiment on alternative node parameter optimization algorithms discused in [our paper](#citation), which we also provide in DiffMat.

* Bayesian optimization (BO): [scikit-optimize](https://scikit-optimize.github.io/stable/), [Ax](https://ax.dev/)
* Simulated annealing: [simanneal](https://github.com/perrygeo/simanneal)

This command will install them altogether.
```bash
pip install scikit-optimize ax-platform simanneal
```

In addition, you may download the [FreeImage](https://imageio.readthedocs.io/en/v2.13.4/reference/_backends/imageio.plugins.freeimage.html) plugin to enable HDR texture images in OpenEXR format (`*.exr`).
```bash
python -c "import imageio; imageio.plugins.freeimage.download()"
```

# Installation

## Substance 3D Automation Tookit

DiffMat uses **sbscooker** and **sbsrender**, two command-line automation tools included in [Adobe Substance 3D Designer](https://www.adobe.com/products/substance3d-designer.html) (AS3D) or [Substance 3D Automation Toolkit](https://helpx.adobe.com/substance-3d-sat.html) (SAT), to pre-cache the output textures of unsupported nodes.

DiffMat assumes that either AS3D or SAT is installed in its default, platform-specific folder and automatically detects the executables inside. A custom install location must be manually specified (see [Getting Started](#getting-started)).

> **NOTE:** Our latest paper claims that DiffMat v2 does not rely on proprietary automation tools to generate noise textures. However, AS3D is still required from a practical perspective since implementing *all* generator nodes in DiffMat entails an unrealistic amount of effort.

> **NOTE:** If you already have AS3D or SAT in your system, we recommend upgrading it to the **latest** version possible (12.1.1 as of 07/01/2022) to avoid any compatibility issue. We will continue to align DiffMat with newer software versions as they emerge.

## Install DiffMat

At the root of the cloned repository, invoke the setup Python script using `pip`:
```bash
pip install .
```
To install in development mode:
```bash
pip install -e .
```
Exit the cloned repo and verify package integrity with:
```bash
cd ..
python -c "import diffmat; print(diffmat.__version__)"
```

# Getting Started

We introduce the two most common ways to employ DiffMat in your procedural material capturing and authoring workflow: **command-line scripts** and **Python API**.

## Command-Line Scripts

For independent usage, we include several Python scripts in the `test/` folder that serve as basic command-line interfaces, including:

* `test_nodes.py` - translates a procedural material graph in `*.sbs` format into a differentiable program and computes output texture maps.
* `test_optimizer.py` - optimizes the *continuous* node parameters inside a procedural material graph to match the material appearance in an input texture image.
* `test_hybrid_optimizer.py` - optimizes both *continuous* and *discrete* node parameters in a procedural material graph.
* `test_sampler.py` - randomly samples or perturbs the *continuous* node parameters of a procedural material graph to generate texture variations.
* `test_exposed_transfer.py` - transfers the definition of an exposed parameter from one graph to another (demo only)

Run each testing script using the following command template. Replace content wrapped by square brackets as needed:

```bash
cd [PATH_TO_DIFFMAT]/test
python test_[NAME].py [PATH_TO_SBS_FILE] [OPTIONS]
```

Below are some shared command-line options across these scripts.

```
Command line options:
    -r PATH         Result directory (where a separate subfolder is created for every translated graph)
    --res INT       Output texture resolution after log2, must be an integer in [0, 12]
    -s SEED         Random seed; the usage varies between scripts
    -t PATH         Custom install location of AS3D or SAT
    -nf FORMAT      Normal format used for rendering output SVBRDF maps ('dx' DirectX or 'gl' OpenGL)
    -e              Force input noise textures to be generated using SAT
    -c              Change the PyTorch device to CPU; otherwise, use the default CUDA device if any
    -l LEVEL        Logging level ('none', 'quiet', 'default', or 'verbose')
```

> **NOTE:** The `-c` option is *mandatory* on systems without a PyTorch-compatible GPU. Please refer to PyTorch documentation for devices accessible via `torch.device('cuda')`.

You may inspect the complete command line options of each script using:

```bash
python test_[NAME].py -h
```

### Result Folder Structure

For each procedural material graph, the output from testing scripts is organized into a folder that bears the same name as the graph. Without any command line options that change output folder names, the default result folder structure looks like:

```
📦result
 ┣ 📦[GRAPH_NAME]
 ┃  ┣ 📂default             Computed SVBRDF maps and physics-based rendering of the source material
 ┃  ┃
 ┃  ┣ 📂external_input
 ┃  ┃ ┗ 📂default           Input noises and patterns to the source material graph
 ┃  ┃
 ┃  ┣ 📂optim_[IMAGE]       Node parameter optimization result against an input texture image
 ┃  ┃ ┣ 📂checkpoints        +- Checkpoint files
 ┃  ┃ ┣ 📂export             +- Exported SBS file after optimization
 ┃  ┃ ┣ 📂render             +- Intermediate renderings
 ┃  ┃ ┣ 📂basecolor          +- Intermediate SVBRDF maps (albedo, normal, roughness, metallic, ...)
 ┃  ┃ ┗ 📂...
 ┃  ┃
 ┃  ┗ 📜summary.yml         Summary of translated material graph structure and node parameter values
 ┃
 ┣ 📦[ANOTHER_GRAPH_NAME]
 ┗ ...
```

## Python API

You can also integrate DiffMat into your Python project via high-level API, which replaces the testing scripts above with equivalent Python classes.

| **Script**                 | **Functionalities**                               | **Python Class**                  |
|----------------------------|---------------------------------------------------|-----------------------------------|
| `test_nodes.py`            | Graph translation & evaluation                    | `diffmat.MaterialGraphTranslator` |
| `test_optimizer.py`        | Gradient-based parameter optimization             | `diffmat.optim.Optimizer`         |
| `test_hybrid_optimizer.py` | Mixed-integer parameter optimization              | `diffmat.optim.HybridOptimizer`   |
| `test_sampler.py`          | Random parameter sampling                         | `diffmat.optim.ParamSampler`      |

For example, the following code snippet translates a procedural material named `wood_american_cherry.sbs` and optimizes graph parameters to match an input photo `wood_dark_brown.jpg`.

> **NOTE:** While we use the `pathlib` package (internal to Python) to create platform-agnostic file paths in this example, ordinary Python strings work as well.

```python
from pathlib import Path

from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.optim import Optimizer
from diffmat.core.io import read_image

# Enable on-screen logging
config_logger(level='default')

# Input and output file paths
sbs_file_path = Path('[PATH_TO_SBS_DIR]') / 'wood_american_cherry.sbs'
img_path = Path('[PATH_TO_IMG_DIR]') / 'wood_dark_brown.jpg'
result_dir = Path('[PATH_TO_RESULT_DIR]')

# Specify a location for storing pre-cached texture images from SAT
external_input_path = result_path / 'external_input'

# Translate the source material graph (using 512x512 resolution)
translator = MGT(sbs_file_path, res=9)
graph = translator.translate(external_input_folder=external_input_path, device='cuda')

# Compile the graph to generate a differentiable program
graph.compile()

# Read the target image (convert into a BxCxHxW tensor) and run gradient-based optimization for 1k iterations
target_img = read_image(img_path)[:3].unsqueeze(0)
optimizer = Optimizer(graph, lr=5e-4)
optimizer.optimize(target_img, num_iters=1000, result_dir=result_dir)
```

> **NOTE:** The documentation to DiffMat Python API currently only consists of docstrings at functions and class methods (see [Code Structure](#code-structure)). We are actively planning on a documentation website for more straightforward navigation and searching capabilities.

## Code Structure

The file tree below illustrates DiffMat's codebase structure, including potential files of interest that might help identify locations for modification or docstring reference.

```
📦diffmat
 ┣ 📂config                 Node type definitions (I/O slots and parameters) and global look-up tables
 ┃  ┣ 📂functions             - Function graph node definitions
 ┃  ┣ 📂fxmap                 - FX-Map graph node definitions
 ┃  ┣ 📂generators            - Noise generator node definitions
 ┃  ┣ 📂nodes                 - Other generator/filter node definitions
 ┃  ┣ 📜function_list.yml     - List of supported function graph nodes
 ┃  ┣ 📜node_list.yml         - List of supported material graph nodes
 ┃  ┗ ...
 ┃
 ┣ 📂core                   Differentiable procedural material graph modules
 ┃  ┣ 📂function              Function graph system
 ┃  ┣ 📂fxmap                 FX-Map graph system
 ┃  ┃  ┣ 📜composer.py          - Simulator of FX-Map nodes with chained Quadrant nodes
 ┃  ┃  ┣ 📜engine_v2.py         - Implementation of the FX-Map engine (calculations behind the scene)
 ┃  ┃  ┣ 📜patterns.py          - Atomic pattern functions in Quadrant node
 ┃  ┃  ┗ ...
 ┃  ┣ 📂material              Material graph system
 ┃  ┃  ┣ 📜functional.py        - Differentiable implementations of atomic and non-atomic nodes
 ┃  ┃  ┣ 📜graph.py             - Material graph class
 ┃  ┃  ┣ 📜node.py              - Material graph node class
 ┃  ┃  ┣ 📜noise.py             - Differentiable implementations of noise/pattern generator nodes
 ┃  ┃  ┣ 📜render.py            - Differentiable physics-based renderer
 ┃  ┃  ┣ 📜param.py             - Node parameter classes
 ┃  ┃  ┗ ...
 ┃  ┣ 📜base.py               - Base classes of all graphs, nodes, and parameters
 ┃  ┣ 📜io.py                 - Image I/O functions; export optimized texture maps to *.sbs
 ┃  ┗ ...
 ┃
 ┣ 📂optim                  Node parameter optimization modules
 ┃  ┣ 📜backend.py            - Implementations of parameter optimization algorithms
 ┃  ┣ 📜descriptor.py         - Texture descriptor class for image feature extraction
 ┃  ┣ 📜metric.py             - Modular loss function
 ┃  ┣ 📜optimizer.py          - Parameter optimization framework definitons
 ┃  ┣ 📜sampler.py            - Random node parameter sampler class
 ┃  ┗ ...
 ┃
 ┗ 📂translator             SBS-to-DiffMat translator modules
    ┣ 📜external_input.py     - Generate input noises and patterns using SAT
    ┣ 📜function_trans.py     - Function graph and node translator classes
    ┣ 📜fxmap_trans.py        - FX-Map graph translator classes
    ┣ 📜graph_trans.py        - Material graph translator class
    ┣ 📜node_trans.py         - Material node translator classes
    ┣ 📜param_trans.py        - Material graph parameter translator classes
    ┗ ...
```

# Limitations

DiffMat bears a few functional limitations as listed below. We will continue to address most (if not all) of them in subsequent releases. Please refer to the [contributing guide](CONTRIBUTING.md) if you are looking forward to any new, exciting feature.

* Only square textures are supported, namely, we assume all intermediate texture maps to have square shapes. Changes in image resolution within a material graph are ignored.
* The source SBS file should only contain one material graph without any custom dependent graphs.
* Unsupported non-atomic material nodes or function nodes. See [FAQs](misc/faqs.md).
* Some rarely used functionalities are omitted in supported nodes. Please refer to the [list of incomplete nodes](incomplete_nodes.md) for more information.
* DiffMat can be heavy in VRAM usage. We recommend that your GPU
have at least **16GB VRAM** if you need to optimize complex, production-grade materials like those presented in our paper.

# FAQs

Check out the dedicated FAQ document [here](misc/faqs.md).

# Citation

We appreciate your citation of the following papers if you find DiffMat useful to your project: [[bibtex]](misc/citation.bib)

> **End-to-End Procedural Material Capture with Proxy-Free Mixed-Integer Optimization** \
> Beichen Li, Liang Shi, Wojciech Matusik \
> *ACM Transactions on Graphics 42(4) (Proc. SIGGRAPH 2023)* \
> [[Paper]](https://dl.acm.org/doi/abs/10.1145/3592132)
>
> **MATch: Differentiable Material Graphs for Procedural Material Capture** \
> Liang Shi, Beichen Li, Miloš Hašan, Kalyan Sunkavalli, Tamy Boubekeur, Radomír Měch, Wojciech Matusik \
> *ACM Transactions on Graphics 39(6) (Proc. SIGGRAPH Asia 2020)* \
> [[Paper]](https://dl.acm.org/doi/abs/10.1145/3414685.3417781) [[Project]](http://match.csail.mit.edu/)

DiffMat has empowered cutting-edge research from computer vision and graphics communities. We list some notable works that incorporate DiffMat to tackle frontier challenges in appearance modeling and inverse rendering:
- [**PhotoScene: Photorealistic Material and Lighting Transfer for Indoor Scenes**](https://yuyingyeh.github.io/projects/photoscene.html), Yeh et al., CVPR 2022.
- [**Node Graph Optimization Using Differentiable Proxies**](https://yiweihu.netlify.app/project/hu2022diff/), Hu et al., SIGGRAPH 2022.
- [**Generating Procedural Materials from Text or Image Prompts**](https://yiweihu.netlify.app/uploads/hu2023gen/project), Hu et al., SIGGRAPH 2023.
- [**PSDR-Room: Single Photo to Scene using Differentiable Rendering**](https://arxiv.org/abs/2307.03244), Yan et al., SIGGRAPH Asia 2023.

Feel free to [email](CONTRIBUTING.md) us if you would like your published work to be acknowledged in the list above.

# License

DiffMat is released under a custom license from MIT and Adobe Inc. Please read our attached [license file](LICENSE) carefully before using the software. We emphasize that DiffMat **shall not** be used for any commercial purposes.
