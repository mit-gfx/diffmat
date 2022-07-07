<div align="center">
  <img width="500px" src="misc/diffmat_logo.png">
</div>
<br/>

[![Latest Release](https://img.shields.io/badge/diffmat-0.1.0-blue)]()

# Overview
***DiffMat*** is a differentiable procedural material modeling library built on [PyTorch](https://pytorch.org). It reproduces the compositing graph system of [Adobe Substance 3D Designer](https://www.adobe.com/products/substance3d-designer.html) with additional auto-differentiation support. DiffMat automatically converts Substance-native procedural materials (in `*.sbs` files) into differentiable computation graphs and optimizes graph parameters against user-captured material appearances (e.g., cellphone flash photos) using a gradient-based method.

# Requirements and Installation

## Python and PyTorch

A minimum of Python 3.7 is required. The PyTorch version used in development is 1.11.0 (with torchvision 0.12.0), and we will test earlier releases to pinpoint the exact minimal version later on.

We provide an example virtual environment configuration file for [Anaconda](https://www.anaconda.com/products/distribution) users, including all dependencies. Clone the DiffMat repository and enter the source folder:
```bash
git clone GIT_URL_TO_DIFFMAT
cd diffmat
```
Create the virtual environment with:
```bash
conda env create -f environment.yml
conda activate diffmat
```

## Substance Designer Command-Line Tools

DiffMat currently does not implement the [FX-Map](https://substance3d.adobe.com/documentation/sddoc/fx-map-172825212.html) node or other noise and pattern generator nodes in Adobe Substance 3D Designer. Thus, it relies on the software's proprietary command-line tools (specifically, **sbscooker** and **sbsrender**) to generate input noises and patterns to a procedural material graph.

If you already have Substance Designer on your system, we recommend upgrading to the **latest** possible version (12.1.1 as of 07/01/2022) to avoid compatibility-related issues. We will continue to align DiffMat with newer versions as they emerge.

DiffMat assumes that Substance Designer is installed in its default, platform-specific folder and automatically detects the required executables inside. You may specify custom install locations using a command-line option to our provided testing scripts or a keyword argument to the material graph translator class (see [Getting Started](#getting-started)).

## Installing DiffMat

In the root folder of the cloned repository, invoke the setup Python script using `pip`:
```bash
pip install .
```
To install in development mode:
```bash
pip install -e .
```
You may optionally download the [FreeImage](https://imageio.readthedocs.io/en/v2.13.4/reference/_backends/imageio.plugins.freeimage.html) plugin to enable HDR image reading and writing in `*.exr` (OpenEXR) format.
```bash
python -c "import imageio; imageio.plugins.freeimage.download()"
```

# Getting Started

We introduce the two most common ways to employ DiffMat in your procedural material capturing and authoring workflow: **command-line scripts** and **Python API**.

## Command-Line Scripts

If you intend to leverage DiffMat as an independent toolset, we provide several Python scripts in the `test/` folder that expose its core functionalities using basic command-line interfaces, including:

* `test_nodes.py` - translates a procedural material graph in Adobe Substance 3D designer (a `*.sbs` document) into a differentiable program and computes output SVBRDF texture maps.
* `test_optimizer.py` -  optimizes the continuous node parameters inside a procedural material graph to match the appearance of an input texture image (either rendered or captured).
* `test_sampler.py` - randomly samples or perturbs the continuous node parameters of a procedural material graph to generate texture variations. The synthetic result may serve as the optimization target in `test_optimizer.py`.
* `test_predictor.py` - trains a parameter prediction neural network to automatically infer node parameters from an input image. The predicted parameters warm-start the optimization process and lead to faster convergence (please see our [paper](#citation) for more details).

Run each testing script using the following command template (remember to drop the square brackets):

```bash
python test_[NAME].py [PATH_TO_SBS_FILE] [OPTIONS]
```

Below are some shared command-line options across these scripts.

```
Command line options:
    -r PATH         Result directory (where a separate subfolder is created for every translated graph)
    --res INT       Output texture resolution after log2, must be an integer in [0, 12]
    -s SEED         Random seed; the usage varies between scripts
    -t PATH         Path to the installed Substance 3D Automation Toolkit
    -nf FORMAT      Normal format used for rendering output SVBRDF maps ('dx' DirectX or 'gl' OpenGL)
    -e              Generate input noises and patterns using SAT (this is currently *required*)
    -c              Change the PyTorch device to CPU; otherwise, the default CUDA device is used
    -l LEVEL        Logging level ('none', 'quiet', 'default', or 'verbose')
```

> Note: If you run DiffMat on a Mac system without an NVIDIA GPU, the `-c` option will be necessary to enforce execution on CPU since PyTorch currently does not support recent Apple GPUs (e.g., M1).

You may access each individual script to acquire a complete command line option list:

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
 ┃  ┣ 📂optim_params_0      Optimization result against a synthetically rendered texture image
 ┃  ┃ ┣ 📂checkpoints        +- Checkpoint files
 ┃  ┃ ┣ 📂export             +- Exported SBS file after optimization
 ┃  ┃ ┣ 📂render             +- Intermediate renderings
 ┃  ┃ ┣ 📂basecolor          +- Intermediate SVBRDF maps (albedo, normal, roughness, metallic, ...)
 ┃  ┃ ┗ 📂...
 ┃  ┃
 ┃  ┣ 📂optim_[IMAGE]       Optimization result against a real-world texture image
 ┃  ┃ ┗ 📂...                +- Identical structure to 'optim_params_0'
 ┃  ┃
 ┃  ┣ 📂sample_default      Random parameter sampling result
 ┃  ┃ ┣ 📂render             +- Sampled renderings
 ┃  ┃ ┣ 📂param              +- Sampled node parameter files
 ┃  ┃ ┣ 📂basecolor          +- Sampled SVBRDF maps
 ┃  ┃ ┗ 📂...
 ┃  ┃
 ┃  ┣ 📂network_train       Result from parameter prediction network training
 ┃  ┃ ┣ 📂checkpoints        +- Network checkpoint files
 ┃  ┃ ┗ 📂validation         +- Comparison between input and predicted textures from validation data
 ┃  ┃
 ┃  ┣ 📂network_pred        Result from parameter prediction network inference
 ┃  ┃ ┗ 📂...                +- Identical structure to 'sample_default'
 ┃  ┃
 ┃  ┗ 📜summary.yml         Summary of translated material graph structure and node parameter values
 ┃
 ┣ 📦[ANOTHER_GRAPH_NAME]
 ┗ ...
```

## Python API

You can also integrate DiffMat into your Python project via high-level API, which replaces the testing scripts above with equivalent Python classes.

| **Script**          | **Functionalities**                               | **Python Class**                  |
|---------------------|---------------------------------------------------|-----------------------------------|
| `test_nodes.py`     | Graph translation & evaluation                    | `diffmat.MaterialGraphTranslator` |
| `test_optimizer.py` | Gradient-based parameter optimization             | `diffmat.optim.Optimizer`         |
| `test_sampler.py`   | Random parameter sampling                         | `diffmat.optim.ParamSampler`      |
| `test_predictor.py` | Parameter prediction network training & inference | `diffmat.optim.ParamPredictor`    |

For example, the following code snippet translates a procedural material named `wood_american_cherry.sbs` and optimizes graph parameters to match an input photo `wood_dark_brown.jpg`.

> Note: While we use the `pathlib` package (internal to Python) to create platform-agnostic file paths in this example, ordinary Python strings work as well.

```python
from pathlib import Path
from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.optim import Optimizer
from diffmat.core.io import read_image

# Enable on-screen logging
config_logger(level='default')

# Input file paths
sbs_file_path = Path('PATH_TO_SBS') / 'wood_american_cherry.sbs'
img_path = Path('PATH_TO_IMG') / 'wood_dark_brown.jpg'

# Specify SAT location and output folders
toolkit_path = Path('PATH_TO_SAT')
result_path = Path('PATH_TO_RESULT')
external_input_path = result_path / 'external_input'

# Translate the source material graph (in 512x512 resolution)
translator = MGT(sbs_file_path, res=9, toolkit_path=toolkit_path)
graph = translator.translate(external_input_folder=external_input_path, device='cuda')

# Compile the graph to generate a differentiable program
graph.compile()

# Read the target image (convert into a BxCxHxW tensor) and run optimization for 1k iterations
target_img = read_image(img_path)[:3].unsqueeze(0)
optimizer = Optimizer(graph, lr=5e-4)
optimizer.optimize(target_img, num_iters=1000, result_dir=result_path)
```

The documentation to DiffMat's Python API currently only consists of docstrings at individual functions and class methods (see [Code Structure](#code-structure)). Nonetheless, we are actively planning on a documentation website for more straightforward navigation and searching capabilities.

## Code Structure

The following list describes the structure of DiffMat's codebase, which might help identify locations of functions and class methods for docstring lookup.

```
📦diffmat
 ┣ 📂config                 Configuration files
 ┃  ┣ 📂functions            +- Function graph node configurations
 ┃  ┣ 📂nodes                +- Material graph node configurations
 ┃  ┣ 📜factory.yml          +- Lookup tables of node/parameter translator classes by category
 ┃  ┣ 📜function_list.yml    +- Supported function graph nodes
 ┃  ┣ 📜node_list.yml        +- Supported material graph nodes
 ┃  ┗ 📜param_types.yml      +- Parameter type specifiers for function graph compilation
 ┃
 ┣ 📂core                   Class definitions and utility functions of differentiable procedural material graphs
 ┃  ┣ 📂export_template      +- Template *.sbs files used for exporting output texture maps
 ┃  ┣ 📜base.py              +- Base classes of parameters, nodes, and graphs
 ┃  ┣ 📜function.py          +- Function graph and node classes
 ┃  ┣ 📜functional.py        +- Differentiable implementations of atomic and non-atomic node functions
 ┃  ┣ 📜graph.py             +- Material graph class
 ┃  ┣ 📜io.py                +- Image I/O functions; export optimized texture maps to *.sbs
 ┃  ┣ 📜log.py               +- Simple logging control
 ┃  ┣ 📜node.py              +- Material node classes
 ┃  ┣ 📜param.py             +- Material graph parameter classes (constant, optimizable, dynamic)
 ┃  ┣ 📜render.py            +- Differentiable physics-based renderer
 ┃  ┣ 📜types.py             +- Typing aliases
 ┃  ┗ 📜util.py              +- Other utility functions (e.g., argument checking in node functions)
 ┃
 ┣ 📂optim                  Related to optimization of node parameters
 ┃  ┣ 📜descriptor.py        +- Texture descriptor class for feature extraction
 ┃  ┣ 📜optimizer.py         +- Parameter optimizer class
 ┃  ┣ 📜predictor.py         +- Neural-network-based parameter predictor class
 ┃  ┗ 📜sampler.py           +- Random parameter sampler class
 ┃
 ┗ 📂translator             Class definitions and utility functions of SBS-to-DiffMat translators
    ┣ 📜base.py              +- Base classes of parameter, node, and graph translators
    ┣ 📜external_input.py    +- Generate input noises and patterns using SAT
    ┣ 📜function_trans.py    +- Function graph and node translator classes
    ┣ 📜graph_trans.py       +- Material graph translator class
    ┣ 📜node_trans.py        +- Material node translator classes
    ┣ 📜param_trans.py       +- Material graph parameter translator classes
    ┣ 📜types.py             +- Typing aliases
    ┗ 📜util.py              +- Other utility functions (e.g., for analyzing parsed XML trees)
```

# Reproducing the MATch Paper (SIGGRAPH Asia 2020)

Please refer to our [step-by-step guide](reproducing_match.md) for reproducing experiment results in the [MATch paper](#citation).

# Limitations

DiffMat is still at an early stage and it has several functional limitations to be aware of. We will continue to address most (if not all) of them in subsequent releases. If you are looking forward to practical additional features, please refer to the [contributing guide](CONTRIBUTING.md).

* Changes in image size within a material graph are ignored.
* Only square textures are supported, namely, we assume all intermediate texture maps to have square shapes.
* Implementations of noise/pattern generator nodes and the generic [FX-Map](https://substance3d.adobe.com/documentation/sddoc/fx-map-172825212.html) node are absent.
* Implementations of [Pixel Processor](https://substance3d.adobe.com/documentation/sddoc/pixel-processor-172825311.html) nodes are not included.
* [Set/Sequence](https://substance3d.adobe.com/documentation/sddoc/using-the-set-sequence-nodes-102400025.html) nodes and non-atomic function nodes are not supported in function graphs.
* The source SBS file should only contain one material graph without any custom dependent graphs.
* Optimization only touches continuous node parameters in a procedural material graph.

# FAQs

- **Q: How is this repository related to the "mit-gfx/diffmat-legacy" repository?** \
  A: The "diffmat-legacy" repository hosts an obsolete version of DiffMat (v0.0.1) which we will no longer maintain. Its sole purpose is for us to fulfill the [license agreement](LICENSE) between MIT and Adobe Inc. We subsequently created the "diffmat" repo under the same license for further development without complicating the legal aspect. Therefore, all future DiffMat releases and related activities will exclusively happen in the "diffmat" repo.

- **Q: Why does the output texture from DiffMat look different from Substance Designer after translation?** \
  A: This should be normal in most cases. While DiffMat thrives to faithfully reproduce the functionalities of atomic and non-atomic nodes in Substance Designer (SD), exact replication is impossible since SD is not open-source by nature. Therefore, any of the following reasons could lead to divergent behaviors between DiffMat and SD:

    - **Stochastic operations in material nodes**. The random number generators used in DiffMat are different from SD as the latter is unknown to the public. Consequently, material nodes that involve stochastic operations will yield statistically similar but not identical results. Some prominent examples are *Safe Transform*, *Make It Tile Patch*, *Dissolve (Blend)*, and function graphs with *Rand* nodes.
    - **Temporarily incomplete node functions**. SD packs abundant features in material nodes but not all of them are for frequent use. Thus, we temporarily omit some rarely occurring functionalities and categroize them in the [list of incomplete nodes](incomplete_nodes.md). Furthermore, there will be latency as we continue to catch up with latest changes in SD. 
    - **Accumulation of numerical errors**. Tiny numerical errors from pixel value quantization and minor differences in node implementation might accumulate and propagate throughout the material graph. Depending on the graph structure, this could result in discrepancies in output texture maps.

  Despite the difficulty in fully reproducing SD's compositing graph system, we generally don't expect a significant mismatch due to graph translation. Please don't hesitate to notify us about any exceptions that you encounter.

- **Q: Should I worry about compatibility issues if my Substance Designer is not the latest version?** \
  A: We highly recommend upgrading to the latest version if that is an option. Otherwise, depending on how far your current version is from the latest one, you could run into varying compatibility issues since SD alters and even revamps material node implementations now and then. In these cases, DiffMat may function but produce slightly different texture maps from SD.

# Citation

DiffMat was initially introduced in the following paper:

> **MATch: Differentiable Material Graphs for Procedural Material Capture** \
> Liang Shi, Beichen Li, Miloš Hašan, Kalyan Sunkavalli, Tamy Boubekeur, Radomír Měch, Wojciech Matusik \
> *ACM Transactions on Graphics 39(6) (Proc. SIGGRAPH Asia 2020)* \
> [[Paper]](https://dl.acm.org/doi/abs/10.1145/3414685.3417781) [[Project]](http://match.csail.mit.edu/)

DiffMat has been used by researchers from computer vision and graphics communities. Here, we list some notable works that incorporate DiffMat to tackle challenges in appearance modeling and inverse rendering:
- [**PhotoScene: Photorealistic Material and Lighting Transfer for Indoor Scenes**](https://yuyingyeh.github.io/projects/photoscene.html), Yeh et al., CVPR 2022.
- [**Node Graph Optimization Using Differentiable Proxies**](https://graphics.cs.yale.edu/publications/node-graph-optimization-using-differentiable-proxies), Hu et al, SIGGRAPH 2022.

If you use DiffMat in your research and find it helpful, please consider citing our paper using the BibTeX entry below. Send us an [email](CONTRIBUTING.md) if you would like your published work to be acknowledged in the list above.
```bibtex
@article{shi2020match,
author = {Shi, Liang and Li, Beichen and Ha\v{s}an, Milo\v{s} and Sunkavalli, Kalyan and Boubekeur, Tamy and Mech, Radomir and Matusik, Wojciech},
title = {MATch: Differentiable Material Graphs for Procedural Material Capture},
year = {2020},
publisher = {Association for Computing Machinery},
volume = {39},
number = {6},
issn = {0730-0301},
articleno = {196},
numpages = {15},
}
```

# License

DiffMat is released under a custom license from MIT and Adobe Inc. Please read our attached [license file](LICENSE) carefully before using the software. We emphasize that DiffMat **shall not** be used for any commercial purposes.
