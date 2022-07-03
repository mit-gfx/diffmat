# Reproducing the MATch Paper (SIGGRAPH Asia 2020)

## Translating a Procedural Material

The following commands translate a procedural material in Substance-native `*.sbs` format into a differentiable program, computes the output SVBRDF maps, and saves them into PNG images in 512x512 resolution. Square brackets must be excluded.
```bash
cd test
python test_nodes.py [PATH_TO_SBS_FILE] -e -b -g -t [TOOLKIT_PATH]
```
Meanings behind command line options:
- The `-e` option requires noises and patterns to be generated from Substance Designer's command-line tools (**sbscooker** and **sbsrender**), with `-t` specifying the installation location of Substance Designer if it is different from the default one.
- The `-b` switch enables backward evaluation of the translated differentiable program, resulting in roughly the same memory consumption as graph parameter optimization. This option tells whether the PyTorch device in use has sufficient memory.
- `-g` prompts DiffMat to generate ground-truth SVBRDF maps alongside its own output for reference. You may compare these texture maps to make sure DiffMat sufficiently reproduces the behavior of Adobe Substance 3D Designer.

## Random Parameter Sampling for Appearance Variations

To sample a procedural material by randomly perturbing its continuous node parameters around their initial values:
```bash
python test_sampler.py [PATH_TO_SBS_FILE] -e -s 0 -t [TOOLKIT_PATH] -ns 5
```
where `-s` specifies the random seed for random parameter sampling, and `-ns` denotes the number of sampled textures.

> The random sampling algorithm is controlled by additional command line options, including the type and range of the random distribution. See `test_sampler.py` for more details.

## Optimization against a Sampled Texture

A randomly sampled procedural material appearance can directly serve as the optimization target. To that end, the name of the sampled graph parameter file should be provided from the command line (e.g., `params_0`).
```bash
python test_optimizer.py [PATH_TO_SBS_FILE] -e -f params_0 -t [TOOLKIT_PATH]
```

## Optimization to Match a Real Image

The command line options differ slightly when designating a real-world captured photo for optimization:
```bash
python test_optimizer.py [PATH_TO_SBS_FILE] -e -im [PATH_TO_IMAGE] -t [TOOLKIT_PATH] --save-output-sbs
```
An extra `--save-output-sbs` switch allows DiffMat to export optimized SVBRDF maps into a local `*.sbs` file.
<!-- Unlike the source material graph, the exported SBS file merely links texture maps using Bitmap nodes and therefore can not accommodate arbitrary resolutions. -->

## Initializing Parameters using a Prediction Network

First of all, train the parameter prediction neural network of a procedural material using:
```bash
python test_predictor.py train [PATH_TO_SBS_FILE] -e -s 0 -v -t [TOOLKIT_PATH]
```
where
- `-v` instructs the script to save intermediate input/prediction image pairs during validation as a means of monitoring network convergence.

> Note: Training a parameter prediction network **in full capacity** consumes a significant amount of GPU memory (more than 16GB). You may add an extra command-line option `-tl 0` to reduce network capacity and constrain GPU usage. However, it is recommended to use a GPU with at least 12GB VRAM.

Next, run the same script in _eval_ mode to infer node parameter values given the optimization target image. Include an epoch number using the `-le` option to load the corresponding checkpoint. The predicted parameter file is saved under the `param/` subfolder of the output directory (`network_pred/` by default).
```bash
python test_predictor.py eval [PATH_TO_SBS_FILE] -e -im [PATH_TO_IMAGE] -le [LOAD_EPOCH] -t [TOOLKIT_PATH]
```

Finally, start optimization from the inferred parameters by supplying an `-ip` option that points to the saved parameter file:
```bash
python test_optimizer.py [PATH_TO_SBS_FILE] -e -im [PATH_TO_IMAGE] -ip [PATH_TO_PARAM_FILE] -t [TOOLKIT_PATH]
```

## Additional Notes for Translating Material Graphs in `test/sbs/match_v1`

The material graphs in `test/sbs/match_v1` were used to generate a gallery figure of 88 procedural materials in the paper.

When translating these graphs using DiffMat's command-line scripts for evaluation or optimization, include an additional option `-nf gl` to change the default normal format to OpenGL. Otherwise, texture renderings might contain artifacts due to inverted normal directions.
