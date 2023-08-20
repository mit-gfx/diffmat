from typing import Union, Tuple, Dict
import math

import torch as th
import taichi as ti

from diffmat.core.types import FloatValue, PatternFunction
from diffmat.core.util import to_const


def square(grid: th.Tensor, _: FloatValue = 0.0) -> Tuple[th.Tensor, th.Tensor]:
    """Generate a square pattern.
    """
    square = ((grid > -1) & (grid <= 1)).all(dim=-1, keepdim=True).float()
    return square, square


def bell(grid: th.Tensor, _: FloatValue = 0.0) -> th.Tensor:
    """Generate a bell pattern.
    """
    x = (grid ** 2).sum(dim=-1, keepdim=True).clamp_max(1.0)
    return (x - 1) ** 4


def brick(grid: th.Tensor, var: FloatValue = 0.0) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
    """Generate a brick pattern.
    """
    # Special case - return a square pattern
    var_const = to_const(var)
    if var_const == 0:
        return square(grid)

    # Compute the pattern by sections - inner: flat; outer: polynomial decay
    grid = grid.clamp(-1.0, 1.0).abs()
    outer_mask = grid > 1 - var_const
    x = (grid - (1 - var)) / (var * 0.5).clamp_min(1e-8) - 1
    outer = 0.25 * x ** 3 - 0.75 * x + 0.5

    one = th.ones([], device=grid.device)
    return th.where(outer_mask, outer, one).prod(dim=-1, keepdim=True)


def brick_batch(grid: th.Tensor, var: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    """Generate brick patterns in a batch.
    """
    # Compute the pattern by sections - inner: flat; outer: polynomial decay
    grid = grid.clamp(-1.0, 1.0).abs()
    outer_mask = grid > 1 - var
    x = (grid - (1 - var)) / (var * 0.5).clamp_min(1e-8) - 1
    outer = 0.25 * x ** 3 - 0.75 * x + 0.5

    # Special case - compute the square mask when var = 1
    square_, _ = square(grid)
    outer = th.where(var == 0, square_, outer)
    one = th.ones([], device=grid.device)
    return th.where(outer_mask, outer, one).prod(dim=-1, keepdim=True), square_


def capsule(grid: th.Tensor, var: FloatValue = 0.0) -> th.Tensor:
    """Generate a capsule pattern.
    """
    x, y = grid.abs().split(1, dim=-1)
    threshold = (1 - math.sqrt(0.05) - 0.006) * var
    d_2 = th.clamp_min(x - threshold, 0.0) ** 2 + y ** 2
    return th.clamp_min(1 - 20 * d_2, 0.0)


def cone(grid: th.Tensor, _: FloatValue = 0.0) -> th.Tensor:
    """Generate a cone pattern.
    """
    return th.clamp_min(1 - grid.norm(dim=-1, keepdim=True), 0.0)


def crescent(grid: th.Tensor, var: FloatValue = 0.0) -> th.Tensor:
    """Generate a crescent pattern.
    """
    # Special case - return black
    var_const = to_const(var)
    if var_const == 1:
        return th.zeros_like(grid.narrow(-1, 0, 1))

    # Create the donut pattern in the center
    d_2 = (grid ** 2).sum(dim=-1, keepdim=True).clamp_max(1.0)
    donut = 6.75 * d_2 * (d_2 - 1) ** 2

    # Create the polynomial decay cover mask
    x = th.clamp_max((grid.narrow(-1, 0, 1) + var) / (1 - var).clamp_min(1e-8), 1.0)
    cover = 0.25 * x ** 3 - 0.75 * x + 0.5
    return donut * cover


def crescent_batch(grid: th.Tensor, var: th.Tensor) -> th.Tensor:
    """Generate crescent patterns in a batch.
    """
    # Create the donut pattern in the center
    d_2 = (grid ** 2).sum(dim=-1, keepdim=True).clamp_max(1.0)
    donut = 6.75 * d_2 * (d_2 - 1) ** 2

    # Create the polynomial decay cover mask
    x = th.clamp_max((grid.narrow(-1, 0, 1) + var) / (1 - var).clamp_min(1e-8), 1.0)
    cover = 0.25 * x ** 3 - 0.75 * x + 0.5

    # Special case - return black when var = 1
    zero = th.zeros([])
    cover = th.where(var == 1, zero, cover)
    return donut * cover


def disc(grid: th.Tensor, _: FloatValue = 0.0) -> Tuple[th.Tensor, th.Tensor]:
    """Generate a disc pattern.
    """
    square_, _ = square(grid)
    return square_ * (grid.norm(dim=-1, keepdim=True) <= 1), square_


def gaussian(grid: th.Tensor, _: FloatValue = 0.0) -> Tuple[th.Tensor, th.Tensor]:
    """Generate a gaussian pattern.
    """
    gaussian = th.exp((grid ** 2).sum(dim=-1, keepdim=True) / -0.09)
    in_pattern = ((grid > -1) & (grid <= 1)).all(dim=-1, keepdim=True)
    return gaussian * in_pattern, in_pattern.float()


def gradation(grid: th.Tensor, var: FloatValue = 0.0) -> Tuple[th.Tensor, th.Tensor]:
    """Generate a linear gradient pattern.
    """
    gradient = (grid.narrow(-1, 0, 1) + 1) * (0.5 * (1 - var))
    in_pattern = ((grid > -1) & (grid <= 1)).all(dim=-1, keepdim=True)
    return gradient * in_pattern, in_pattern.float()


def gradation_offset(grid: th.Tensor, var: FloatValue = 0.0) -> Tuple[th.Tensor, th.Tensor]:
    """Generate a linear gradient pattern with offset.
    TODO: fix color behavior.
    """
    gradient_offset = (grid.narrow(-1, 0, 1) + var * 2 + 1) * 0.5
    in_pattern = ((grid > -1) & (grid <= 1)).all(dim=-1, keepdim=True)
    return gradient_offset * in_pattern, in_pattern.float()


def half_bell(grid: th.Tensor, var: FloatValue = 0.0) -> th.Tensor:
    """Generate a half bell pattern.
    """
    x, bell_ = grid.narrow(-1, 0, 1), bell(grid)

    # Apply a polynomial decay on the right half
    if to_const(var) == 1:
        half_bell = bell_ * (x > 0)
    else:
        x = th.clamp_max(x / (1 - var).clamp_min(1e-8), 1.0)
        decay_bell = bell_ * (2 * x ** 3 - 3 * x ** 2 + 1)
        half_bell = th.where(x > 0, decay_bell, bell_)
    return half_bell


def half_bell_batch(grid: th.Tensor, var: FloatValue = 0.0) -> th.Tensor:
    """Generate half bell patterns in a batch.
    """
    x, bell_ = grid.narrow(-1, 0, 1), bell(grid)

    # Apply a polynomial decay on the right half
    x = th.clamp_max(x / (1 - var).clamp_min(1e-8), 1.0)
    decay_bell = bell_ * (2 * x ** 3 - 3 * x ** 2 + 1)

    # Special case - the right half is black when var = 1
    zero = th.zeros([])
    decay_bell = th.where(var == 1, zero, decay_bell)
    half_bell = th.where(x > 0, decay_bell, bell_)
    return half_bell


def paraboloid(grid: th.Tensor, _: FloatValue = 0.0) -> th.Tensor:
    """Generate a paraboloid pattern.
    """
    return th.clamp_min(1 - (grid ** 2).sum(dim=-1, keepdim=True), 0.0)


def pyramid(grid: th.Tensor, _: FloatValue = 0.0) -> th.Tensor:
    """Generate a pyramid pattern.
    """
    return th.clamp_min(1 - grid.abs().max(dim=-1, keepdim=True)[0], 0.0)


def ridged_bell(grid: th.Tensor, _: FloatValue = 0.0) -> th.Tensor:
    """Generate a ridged bell pattern.
    """
    x, bell_ = grid.narrow(-1, 0, 1), bell(grid)
    return (x.abs().clamp_max(1.0) - 1) ** 2 * bell_


def thorn(grid: th.Tensor, _: FloatValue = 0.0) -> th.Tensor:
    """Generate a thorn pattern.
    """
    return (grid.norm(dim=-1, keepdim=True).clamp_max(1.0) - 1) ** 4


def waves(grid: th.Tensor, var: FloatValue = 0.0) -> th.Tensor:
    """Generate a waves pattern.
    """
    paraboloid_ = paraboloid(grid)
    coeff = (var * 12 + 3) * math.pi
    raised_cosine = (th.cos(grid.narrow(-1, 0, 1) * coeff) + 1) * 0.5
    return paraboloid_ * raised_cosine


# Dictionary of all atomic patterns
ATOMIC_PATTERNS: Dict[str, PatternFunction] = {
    'square': square,
    'disc': disc,
    'paraboloid': paraboloid,
    'bell': bell,
    'gaussian': gaussian,
    'thorn': thorn,
    'pyramid': pyramid,
    'brick': brick,
    'gradation': gradation,
    'waves': waves,
    'half_bell': half_bell,
    'ridged_bell': ridged_bell,
    'crescent': crescent,
    'capsule': capsule,
    'cone': cone,
    'gradation_offset': gradation_offset
}

# Dictionary of batched versions of atomic patterns
ATOMIC_PATTERNS_BATCH = ATOMIC_PATTERNS.copy()
ATOMIC_PATTERNS_BATCH.update({
    'brick': brick_batch,
    'crescent': crescent_batch,
    'half_bell': half_bell_batch
})


# ----------------------------- #
#        Taichi routines        #
# ----------------------------- #

@ti.func
def square_ti(x: float, y: float) -> float:
    """Square pattern (Taichi version).
    """
    return 1.0 if x > -1 and y > -1 and x <= 1 and y <= 1 else 0.0


@ti.func
def bell_ti(x: float, y: float) -> float:
    """Bell pattern (Taichi version).
    """
    return (ti.min(x ** 2 + y ** 2, 1.0) - 1) ** 4


@ti.func
def brick_weight_ti(x: float, var: float) -> float:
    """Helper function for brick pattern (Taichi version).
    """
    a = (x - (1 - var)) * 2 / var - 1
    return 0.25 * a ** 3 - 0.75 * a + 0.5


@ti.func
def brick_ti(x: float, y: float, var: float) -> float:
    """Brick pattern (Taichi version).
    """
    x, y = ti.min(abs(x), 1.0), ti.min(abs(y), 1.0)
    wx = 1.0 if x <= 1 - var else brick_weight_ti(x, var)
    wy = 1.0 if y <= 1 - var else brick_weight_ti(y, var)
    return wx * wy


@ti.func
def capsule_ti(x: float, y: float, var: float) -> float:
    """Capsule pattern (Taichi version).
    """
    t = (1 - ti.sqrt(0.05) - 0.006) * ti.max(var, 0.0)
    d_sq = ti.max(abs(x) - t, 0.0) ** 2 + y ** 2
    return ti.max(1 - 20 * d_sq, 0.0)


@ti.func
def cone_ti(x: float, y: float) -> float:
    """Cone pattern (Taichi version).
    """
    return ti.max(1 - ti.sqrt(x ** 2 + y ** 2), 0.0)


@ti.func
def crescent_ti(x: float, y: float, var: float) -> float:
    """Crescent pattern (Taichi version).
    """
    pixel = 0.0

    if var >= 1.0:
        pixel = 0.0
    else:
        d_sq = ti.min(x ** 2 + y ** 2, 1.0)
        donut = 6.75 * d_sq * (d_sq - 1) ** 2
        a = ti.min((x + var) / (1 - var), 1.0)
        cover = 0.25 * a ** 3 - 0.75 * a + 0.5
        pixel = donut * cover

    return pixel


@ti.func
def disc_ti(x: float, y: float) -> float:
    """Disc pattern (Taichi version).
    """
    return square_ti(x, y) if ti.sqrt(x ** 2 + y ** 2) <= 1 else 0.0


@ti.func
def gaussian_ti(x: float, y: float) -> float:
    """Gaussian pattern (Taichi version).
    """
    g = ti.exp((x ** 2 + y ** 2) / -0.09)
    return square_ti(x, y) * g


@ti.func
def gradation_ti(x: float, y: float, var: float) -> float:
    """Gradation pattern (Taichi version).
    """
    g = (x + 1) * 0.5 * ti.max(1 - var, 0.0)
    return square_ti(x, y) * g


@ti.func
def gradation_offset_ti(x: float, y: float, var: float) -> float:
    """Gradation w/ offset pattern (Taichi version).
    TODO: fix color behavior.
    """
    g = (x + 1) * 0.5 + var
    return square_ti(x, y) * g


@ti.func
def half_bell_ti(x: float, y: float, var: float) -> float:
    """Half bell pattern (Taichi version).
    """
    pixel = 0.0

    if var >= 1.0:
        pixel = bell_ti(x, y) if x > 0 else 0.0
    else:
        a = ti.min(x / ti.min(1 - var, 1.0), 1.0)
        w = 2 * a ** 3 - 3 * a ** 2 + 1 if x > 0 else 1.0
        pixel = bell_ti(x, y) * w

    return pixel


@ti.func
def paraboloid_ti(x: float, y: float) -> float:
    """Paraboloid pattern (Taichi version).
    """
    return ti.max(1 - x ** 2 - y ** 2, 0.0)


@ti.func
def pyramid_ti(x: float, y: float) -> float:
    """Pyramid pattern (Taichi version).
    """
    return ti.max(1 - ti.max(abs(x), abs(y)), 0.0)


@ti.func
def ridged_bell_ti(x: float, y: float) -> float:
    """Ridged bell pattern (Taichi version).
    """
    r = (ti.min(abs(x), 1.0) - 1) ** 2
    return bell_ti(x, y) * r


@ti.func
def thorn_ti(x: float, y: float) -> float:
    """Thorn pattern (Taichi version).
    """
    return (ti.min(ti.sqrt(x ** 2 + y ** 2), 1.0) - 1) ** 4


@ti.func
def waves_ti(x: float, y: float, var: float) -> float:
    """Waves pattern (Taichi version).
    """
    c = (ti.max(var, 0.0) * 12 + 3) * math.pi
    w = (ti.cos(x * c) + 1) * 0.5
    return paraboloid_ti(x, y) * w


@ti.func
def pattern_ti(x: float, y: float, var: float, type: int) -> float:
    """Aggregated atomic pattern function (Taichi version).
    """
    pixel = 0.0

    if type == 0:
        pixel = square_ti(x, y)
    elif type == 1:
        pixel = disc_ti(x, y)
    elif type == 2:
        pixel = paraboloid_ti(x, y)
    elif type == 3:
        pixel = bell_ti(x, y)
    elif type == 4:
        pixel = gaussian_ti(x, y)
    elif type == 5:
        pixel = thorn_ti(x, y)
    elif type == 6:
        pixel = pyramid_ti(x, y)
    elif type == 7:
        pixel = brick_ti(x, y, var)
    elif type == 8:
        pixel = gradation_ti(x, y, var)
    elif type == 9:
        pixel = waves_ti(x, y, var)
    elif type == 10:
        pixel = half_bell_ti(x, y, var)
    elif type == 11:
        pixel = ridged_bell_ti(x, y)
    elif type == 12:
        pixel = crescent_ti(x, y, var)
    elif type == 13:
        pixel = capsule_ti(x, y, var)
    elif type == 14:
        pixel = cone_ti(x, y)
    elif type == 15:
        pixel = gradation_offset_ti(x, y, var)

    return pixel
