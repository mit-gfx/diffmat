from abc import abstractmethod
from typing import List, Dict, Callable, Type, Optional, Any

from torch.fft import rfft2
import torch as th
import torch.nn.functional as F

from diffmat.core.base import BaseEvaluableObject
from diffmat.core.material.util import input_check
from diffmat.optim.descriptor import TextureDescriptor


class BaseMetric(BaseEvaluableObject):
    """Base class for loss functions to be used in parameter optimizers.
    """
    def __init__(self, loss_type: str = 'l1', td_kwargs: Dict[str, Any] = {}, **kwargs):
        """Initialize the loss function module.
        """
        super().__init__(**kwargs)

        # Initialize the target texture descriptor
        self.target_td: Optional[th.Tensor] = None
        self.td_kwargs = td_kwargs

        # Retrieve the loss function
        if loss_type not in LOSS_DICT:
            raise ValueError(f'Unknown loss function type: {loss_type}. Valid options are '
                             f'{list(LOSS_DICT.keys())}.')
        else:
            self.loss_func = LOSS_DICT[loss_type]

    @abstractmethod
    def _calc_descriptor(self, img: th.Tensor) -> th.Tensor:
        """Compute the texture descriptor for a batch of input images (BxCxHxW).
        """
        ...

    @input_check(1, class_method=True)
    def set_target_image(self, img: th.Tensor):
        """Compute and store the texture descriptor for a batch of target images (BxCxHxW).
        """
        self.target_td = self._calc_descriptor(img)

    @input_check(1, class_method=True)
    def evaluate(self, img: th.Tensor) -> th.Tensor:
        """Compute the loss function against the target image.
        """
        # Verify that a target image has been designated
        if self.target_td is None:
            raise RuntimeError('A target image has not been designated')

        # Return the loss value
        return self.loss_func(self._calc_descriptor(img), self.target_td)

    # Shortcut to loss function evaluation
    __call__ = evaluate


class VGGMetric(BaseMetric):
    """Loss function where the texture descriptor comes from VGG19 features.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the VGG loss function.
        """
        super().__init__(*args, **kwargs)

        # Instantiate the texture descriptor module
        self.td = TextureDescriptor(parent=self, device=self.device)

    def _calc_descriptor(self, img: th.Tensor) -> th.Tensor:
        """Compute the VGG19 texture descriptor for a batch of input images (BxCxHxW).
        """
        return self.td.evaluate(img, **self.td_kwargs)


class FFTMetric(BaseMetric):
    """Loss function where the texture descriptor comes from 2D Fourier transform. 
    """
    def __init__(self, *args, **kwargs):
        """Initialize the FFT loss function.
        """
        super().__init__(*args, **kwargs)

    def _calc_descriptor(self, img: th.Tensor) -> th.Tensor:
        """Compute the FFT texture descriptor for a batch of input images (BxCxHxW).
        """
        return rfft2(img, **self.td_kwargs).abs()


class CombinedMetric(BaseMetric):
    """Loss function that combines multiple features.
    """
    def __init__(self, loss_type: str = 'l1', config: Dict[str, Dict[str, Any]] = {}, **kwargs):
        """Initialize the combined loss function.
        """
        super().__init__(loss_type=loss_type, **kwargs)
        kwargs.pop('parent', None)

        # Generate a list of metrics from the provided configuration
        if config:
            tds: List[BaseMetric] = []
            weights: List[float] = []

            for key, val in config.copy().items():
                weights.append(val.pop('weight', 1.0))
                tds.append(get_metric(key, **val, **kwargs, parent=self))

        # Use a default configuration
        else:
            tds = [get_metric(name, **kwargs, parent=self) for name in ('vgg', 'fft')]
            weights = [1.0, 1e-3]

        self.tds, self.weights = tds, weights

        # Redefine the target texture descriptor as a list of TDs from each loss component
        self.target_td: List[Optional[th.Tensor]] = [None] * len(tds)

    def _calc_descriptor(self, img: th.Tensor) -> List[th.Tensor]:
        """Compute the combined texture descriptor for a batch of input images (BxCxHxW).
        """
        return [td._calc_descriptor(img) for td in self.tds]

    @input_check(1, class_method=True)
    def evaluate(self, img: th.Tensor) -> th.Tensor:
        """Compute the loss function against the target image. The loss is a weighted combination
        of contributions from loss components.
        """
        return sum(self.loss_func(img_td, gt_td) * weight for img_td, gt_td, weight in
                   zip(self._calc_descriptor(img), self.target_td, self.weights))

    __call__ = evaluate


def get_metric(name: str, *args, **kwargs) -> BaseMetric:
    """Create a loss function module by name. Other keyword arguments are passed to the
    constructor of the loss function.
    """
    if name not in METRIC_DICT:
        raise ValueError(f'Unknown loss function name: {name}. Valid options are '
                         f'{list(METRIC_DICT.keys())}.')

    return METRIC_DICT[name](*args, **kwargs)


# Dictionary of loss functions
LOSS_DICT: Dict[str, Callable[[th.Tensor, th.Tensor], th.Tensor]] = {
    'l1': F.l1_loss,
    'l2': F.mse_loss
}

# Dictionary of metrics
METRIC_DICT: Dict[str, Type[BaseMetric]] = {
    'vgg': VGGMetric,
    'fft': FFTMetric,
    'combine': CombinedMetric,
}
