# FAQs

## Troubleshooting

### Q: DiffMat reports "Node configuration of '...' does not exist".

This is a common info message indicating that the source `*.sbs` file contains material nodes not supported by DiffMat as the configuration files for translating these nodes do not exist. These unsupported nodes and their preceding subgraphs will be frozen during optimization, whose output textures are pre-cached using Substance Designer CLI tools.

Due to a vast spectrum of material node types in AS3D, we will try our best to implement additional, commonly used material nodes as needed. Please feel free to open a feature request if you have specific recommendations.

### Q: DiffMat reports "Node configuration of '...' does not have input/output connector '...'".

This info message typically occurs when the source material graph contains legacy node types (e.g., "*Tile Generator (Legacy)*" and "*Clouds 2 (Legacy)*") whose input and output slots are defined differently from their current versions. These legacy nodes are treated as unsupported and their outputs are pre-cached before optimization. Bypassing this message requires manually replacing legacy nodes with current versions in AS3D.

Alternatively, it is possible that AS3D has updated the implementation of some node types in newer releases. Feel free to [send us a reminder](../CONTRIBUTING.md) if DiffMat needs to be updated accordingly.

### Q: Why does the translated material graph generate different texture maps from AS3D?

While we aim to faithfully reproduce the AS3D's procedural material graph system with DiffMat, an exact replication is *impossible* due to the proprietary nature of AS3D. Nonetheless, the visual difference between materials generated using DiffMat and AS3D should be minor in most cases. Please open an issue if you notice a substantial deviation in the output texture after graph translation.

Below are some possible reasons for mismatches between DiffMat and AS3D:
* **Randomness in material nodes.** The random number generators used in DiffMat aren't open-source. Therefore, material nodes that implement any degree of randomness might yield statistically similar but not identical results in DiffMat. Some prominent examples are *Safe Transform*, *Make It Tile Patch*, *Dissolve (Blend)*, and function graphs with *Rand* nodes.
* **Temporarily incomplete node functions.** AS3D packs abundant features in material nodes but not all of them are used frequently. Thus, we temporarily omit a few rarely occurring parameters as described in the list of incomplete nodes.
* **Accumulation of numerical errors.** Minor numerical differences from individual nodes might accumulate and propagate throughout the material graph.

### Q: Should I worry about compatibility issues if my AS3D is not at the latest version?
AS3D might refresh the implementation of certain material nodes in a major release. The obsolete versions are typically flagged as "*(Legacy)*" node types. DiffMat will drop the support of legacy node types once catching up with the latest AS3D release. We therefore highly recommend upgrading AS3D to the latest version.

## Logistics

### Q: Can I obtain a commercial license to DiffMat?

Tagging Issue [#2](https://github.com/mit-gfx/diffmat/issues/2). Adobe owns an exclusive commercial license to DiffMat besides the current research-only license. Any request for commercial use should thus be directed to Adobe's legal department. You may consider emailing [Tamy Boubekeur](https://perso.telecom-paristech.fr/boubek/), a senior director at Adobe Research who leads Substance-related research teams, and asking for a point of contact.

### Q: How is this repository related to the "mit-gfx/diffmat-legacy" repository?

The "[diffmat-legacy](https://github.com/mit-gfx/diffmat-legacy)" repository hosts an obsolete version of DiffMat (v0.0.1) which is no longer maintained. Its sole purpose is to fulfill the license agreement between MIT and Adobe. We subsequently created the "diffmat" repo under the same license for further development without complicating the legal aspect. Therefore, all future DiffMat releases and other activities will take place in the "diffmat" repo exclusively.
