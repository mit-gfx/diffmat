0.1.0 (06/30/2022)
==================

Initial Commit


0.2.0 (08/20/2022)
==================

### New features
- Add differentiable noise generator nodes
- Support Pixel Processor and FX-map nodes
- Support joint optimization of continuous and integer parameters
- Support non-atomic function nodes
- Introduce exposed parameter transfer as a tech demo

### Enhancement
- Improve runtime in various filter nodes (particularly gradient map nodes)
- Restructure codebase

### Bug Fixes
- Automatically rerun `sbsrender` using CPU SSE2 engine if the default engine fails to generate texture output (Issue [#1](https://github.com/mit-gfx/diffmat/issues/1))