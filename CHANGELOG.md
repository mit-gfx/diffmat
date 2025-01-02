0.1.0 (06/30/2022)
==================

Initial Commit


0.2.0 (08/20/2023)
==================

### New Features
- Add differentiable noise generator nodes
- Support Pixel Processor and FX-map nodes
- Support joint optimization of continuous and integer parameters
- Support non-atomic function nodes

### Enhancements
- Improve the runtime of various filter nodes (particularly gradient map nodes)
- Restructure codebase

### Bug Fixes
- Automatically rerun `sbsrender` using CPU SSE2 engine if the default engine fails to generate texture output (Issue [#1](https://github.com/mit-gfx/diffmat/issues/1))


0.2.1 (01/01/2025)
==================

### New features
- Support LPIPS and L\*a\*b\* color space metrics as optimization objectives
- Support line search optimizers
- Add control over the range of saved data during optimization

### Enhancements
- Any unsupported/legacy material node types and their preceding subgraphs are frozen during optimization instead of triggering translation errors
- Intermediate results during the forward evaluation of material graphs are discarded by default to reduce VRAM consumption
- Material nodes now actively report errors on missing required input connections

### Bug Fixes
- Fix the copying of custom dependencies before generating external input textures using SAT
- Correct the translation of certain unsigned integers into negative integers
- Miscellaneous minor fixes in material and function node implementations
