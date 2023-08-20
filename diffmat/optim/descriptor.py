from typing import List

from torchvision.models.vgg import vgg19
from torch.nn.functional import interpolate
import torch as th
import torch.nn as nn

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.types import DeviceType
from diffmat.core.material.util import input_check


class TextureDescriptor(BaseEvaluableObject):
    """Texture descriptor evaluation based on a pretrained VGG19 network.
    """
    def __init__(self, device: DeviceType = 'cpu', **kwargs):
        """Initialize the texture descriptor evaluator.

        Args:
            device (DeviceType, optional): Device placement of the texture descriptor network.
                Defaults to 'cpu'.
            kwargs (Dict[str, Any], optional): Keyword arguments to pass into the parent class
                constructor.
        """
        super().__init__(device=device, **kwargs)

        # Record intermediate results from the feature extraction network to compute the texture
        # descriptor
        self.features: List[th.Tensor] = []

        # Set up the feature extraction network
        self._setup_model()

        # Image statistics for normalizing an input texture
        self.mean = th.tensor([0.485, 0.456, 0.406], device=self.device).view(-1, 1, 1)
        self.std = th.tensor([0.229, 0.224, 0.225], device=self.device).view(-1, 1, 1)

    def _setup_model(self):
        """Initialize the texture feature extraction model.
        """
        # Get a pretrained VGG19 network and set it to evaluation state
        model: nn.Sequential = vgg19(pretrained=True).features.to(self.device)
        model.eval()

        # Disable network training
        for param in model.parameters():
            param.requires_grad_(False)

        # Change the max pooling to average pooling
        for i, module in enumerate(model):
            if isinstance(module, nn.MaxPool2d):
                model[i] = nn.AvgPool2d(kernel_size=2)

        # The forward hook function for capturing output at a certain network layer
        def forward_hook(module: nn.Module, input: th.Tensor, output: th.Tensor):
            self.features.append(output)

        # Register the forward hook function
        for i in (4, 9, 18, 27):
            model[i].register_forward_hook(forward_hook)

        self.model = model

    def _texture_descriptor(self, img: th.Tensor) -> th.Tensor:
        """Compute the texture descriptor of an input image of shape `(B, C, H, W)`.

        Args:
            img (Tensor): A mini-batch of images.

        Returns:
            Tensor: Texture descriptors of input images, in a shape of `(B, feature_size)`.
        """
        # Normalize the input image
        img = (img - self.mean) / self.std

        # Run the VGG feature extraction network
        self.features.clear()
        self.features.append(self.model(img))

        def gram_matrix(img_feature: th.Tensor) -> th.Tensor:
            mat = img_feature.flatten(-2)
            gram = th.matmul(mat, mat.transpose(-2, -1)) / mat.shape[-1]
            return gram.flatten(1)

        # Compute the Gram matrices using recorded features
        # The feature descriptor has a shape of (B, F), where F is feature length
        return th.cat([gram_matrix(img_feature) for img_feature in self.features], dim=1)

    @input_check(1, class_method=True)
    def evaluate(self, img: th.Tensor, td_level: int = 2) -> th.Tensor:
        """Compute the texture descriptor of an input image at multiple scales.

        Args:
            img (Tensor): A mini-batch of images whose shape is `(B, C, H, W)`.
            td_level (int, optional): Mipmap stack levels when calculating the texture
                descriptor at multiple scales. The feature vector size grows proportionally with
                this number. Defaults to 2.

        Raises:
            ValueError: Texture descriptor level is not an integer or holds a negative value.

        Returns:
            Tensor: Multi-scale texture descriptors of input images, with a shape of
                `(B, feature_size * (td_level + 1))`.
        """
        if not isinstance(td_level, int) or td_level < 0:
            raise ValueError('The texture descriptor level must be a non-negative integer')

        with self.timer('Texture descriptor', log_level='debug'):

            # Compute the texture descriptor at native resolution
            img = img.contiguous()
            tds: List[th.Tensor] = [self._texture_descriptor(img)]

            # Repeat for downscaled images
            for level in range(1, td_level + 1):
                img_scaled = interpolate(
                    img, scale_factor=2**(-level), mode='bilinear', align_corners=True)
                tds.append(self._texture_descriptor(img_scaled))

            td_output = th.cat(tds, dim=1)

        return td_output
