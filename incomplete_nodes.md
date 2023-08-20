# List of Incomplete Material Node Functionalities

Below is a work-in-progress list of Substance Designer's material node functionalities unimplemented by DiffMat. The nodes in question can still be successfully translated but the involved parameters and features are ignored.

## Blend

- Blending mode "Straight alpha blending" and "Premultiplied alpha blending (copy only)".

## Curve

- Per-channel curve adjustment (R, G, B, A). DiffMat currently maps all channels together using a unified curve as if the input image were grayscale.

## Distance

- Distance transforms using "Manhattan" and "Chebyshev" distances.

## Gradient Map

- Gradient interpolation mode "Smooth" with a "Smoothness" parameter.

## Bevel

- The "Corner Type" parameter ("Round" or "Angular").
- The "Use Custom Curve" switch that enables an optional height curve input slot.

## Quantize

- The "Offset", "Slope" and "Slope Curve" options, which have just been introduced in the latest SD update.
- Following the said update, the *Quantize* node is now implemented by a *Pixel Processor* and behaves somewhat differently from previous versions.
