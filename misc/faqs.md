# FAQs

## Troubleshooting

### Q: DiffMat reports "FileNotFoundError: Configuration file not found for node type: ...".

This is a common error indicating that the source procedural material graph contains material or function nodes not supported by DiffMat, which causes failure in graph translation. Unfortunately, AS3D defines much more material node types than we can possibly reproduce. While we try our best to include additional, commonly used material and function nodes, please open a feature request if you recommend us to implement any crucial node type.

### Q: What does "KeyError: ..." mean from "MaterialGraphTranslator._init_graph_connectivity"?

This error typically occurs when the source material graph contains legacy node types such as "Tile Generator (Legacy)" and "Clouds 2 (Legacy)", whose input and output slots are named differently from their latest versions. There are two potential fixes:
* Manually replace legacy nodes with non-legacy counterparts in AS3D.
* Generate the output of these nodes from SAT using the `-e` command line option. This only works if the legacy node is functionally a noise generator, i.e., it does not receive input from other nodes.

Alternatively, it might be the case that AS3D has changed the implementations of some node types with newer releases. Feel free to [send us a reminder](../CONTRIBUTING.md) if DiffMat needs to be updated accordingly.


### Q: Why does the output texture from DiffMat look different from AS3D after graph translation?

While DiffMat thrives to faithfully reproduce the functionalities of atomic and non-atomic nodes in AS3D, exact replication is *impossible* as AS3D is proprietary software. Nonetheless, the behavioral difference between DiffMat and AS3D is mostly minor. Please don't hesitate to notify us if you notice substantial deviation in the output texture due to graph translation.

Below are some possible reasons for mismatch between DiffMat and SD:
* **Randomness in material nodes.** The random number generators used in DiffMat are different from SD as the latter is non-open-source. Consequently, material nodes that implement randomness will yield statistically similar but not identical results. Some prominent examples are *Safe Transform*, *Make It Tile Patch*, *Dissolve (Blend)*, and function graphs with *Rand* nodes.
* **Temporarily incomplete node functions.** SD packs abundant features in material nodes but not all of them are for frequent use. Thus, we temporarily omit some rarely occurring functionalities and categroize them in the list of incomplete nodes. Furthermore, there will be latency as we continue to catch up with latest changes in SD.
* **Accumulation of numerical errors.** Tiny numerical errors from pixel value quantization and minor differences in node implementation might accumulate and propagate throughout the material graph.

### Q: Should I worry about compatibility issues if my AS3D is not at the latest version?
We highly recommend upgrading to the latest version if that is viable. Otherwise, you could run into compatibility issues to a varying degree depending on how far your current version is from the latest because AS3D might alter or even revamp material node implementations with new releases. In such cases, DiffMat may function but produce different texture maps from AS3D.

## Logistics

### Q: How do I obtain a commercial license to DiffMat?

Tagging Issue [#2](https://github.com/mit-gfx/diffmat/issues/2). Besides the current research-only license, Adobe also owns an exclusive commercial license to DiffMat. Any request for commercial use should be directed to Adobe's legal team. You may consider sending an email to [Tamy Boubekeur](https://perso.telecom-paristech.fr/boubek/), a senior director at Adobe Research who leads research efforts around Adobe Substance 3D products, and asking for a point of contact.

### Q: How is this repository related to the "mit-gfx/diffmat-legacy" repository?

The "[diffmat-legacy](https://github.com/mit-gfx/diffmat-legacy)" repository hosts an obsolete version of DiffMat (v0.0.1) that is no longer under maintenance. Its sole purpose is to fulfill the license agreement between MIT and Adobe Inc. We subsequently created the "diffmat" repo under the same license for further development without complicating the legal aspect. Therefore, all future DiffMat releases and related activities will take place in the "diffmat" repo exclusively.
