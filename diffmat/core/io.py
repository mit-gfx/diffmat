from xml.etree import ElementTree as ET
from pathlib import Path, PurePath
from typing import Dict
import math

import imageio
import torch as th
import numpy as np

from diffmat.core.log import get_logger
from diffmat.core.types import PathLike, DeviceType
from diffmat.core.util import check_output_dict


# Logger for the io module
logger = get_logger('diffmat.core')


def read_image(filename: PathLike, device: DeviceType = 'cpu') -> th.Tensor:
    """Read a local image file into a float tensor (pixel values are normalized to [0, 1]). The
    read image is rearranged in CxHxW format.

    Args:
        filename (PathLike): Image file path.
        device (DeviceType, optional): Target device placement of the image. Defaults to 'cpu'.

    Raises:
        ValueError: The input image has a pixel width other than 8-bit or 16-bit.

    Returns:
        Tensor: Loaded image in floating-point tensor format (pixels normalized to [0, 1]).
    """
    img_np: np.ndarray = imageio.imread(filename)

    # Convert the image array to float tensor according to its data type
    if img_np.dtype == np.uint8:
        img_np = img_np.astype(np.float32) / 255.0
    elif img_np.dtype == np.uint16:
        img_np = img_np.astype(np.float32) / 65535.0
    else:
        raise ValueError(f'Unrecognized image pixel value type: {img_np.dtype}')

    # For grayscale images, prepend an extra dimension
    # For color images, convert the image from HWC to CHW
    if img_np.ndim < 3:
        return th.from_numpy(img_np).unsqueeze(0).to(device)
    else:
        return th.from_numpy(img_np).movedim(2, 0).to(device)


def write_image(img: th.Tensor, filename: PathLike, img_format: str = 'png'):
    """Write a CxHxW float tensor into a image file.

    Args:
        img (Tensor): Source image tensor.
        filename (PathLike): Output file path.
        img_format (str, optional): Image file format ('png' or 'exr'). Defaults to 'png'.

    Raises:
        ValueError: The image format is neither 'png' nor 'exr'.
        ValueError: The source image is not a 2D or 3D floating-point tensor.
    """
    # Check input validity
    if img_format not in ('png', 'exr'):
        raise ValueError("The output image format must be either 'png' or 'exr'")
    if not isinstance(img, th.Tensor) or not img.is_floating_point() or img.ndim not in (2, 3):
        raise ValueError('The source image must be a 2D or 3D floating-point tensor')

    # Convert the image back to HxWxC NumPy array
    if img.ndim == 2:
        img = img.detach()
    elif img.shape[0] > 1:
        img = img.detach().movedim(0, 2)
    else:
        img = img.detach().squeeze(0)
    img_np: np.ndarray = img.cpu().numpy().astype(np.float32)

    # For png images, save the array as 8-bit or 16-bit integers
    # For exr images, directly save the float array
    if img_format == 'png':
        if img_np.ndim == 3:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = (img_np * 65535).astype(np.uint16)

    filename = PurePath(filename).with_suffix(f'.{img_format}')
    imageio.imwrite(str(filename), img_np)


def load_input_dict(img_folder: PathLike, glob_pattern: str = '*.*',
                    device: DeviceType = 'cpu') -> Dict[str, th.Tensor]:
    """Load the images in a folder into a dictionary with file names as keys. Images are stored as
    4D tensors in BxCxHxW format.

    Args:
        img_folder (PathLike): Source directory path.
        glob_pattern (str, optional): Pattern used for filtering image files. Defaults to '*.*'.
        device (DeviceType, optional): Target device ID for all loaded images. Defaults to 'cpu'.

    Returns:
        Dict[str, Tensor]: Loaded image dictionary. The keys are extracted from image file names.
    """
    input_dict: Dict[str, th.Tensor] = {}
    for filename in Path(img_folder).glob(glob_pattern):
        input_dict[filename.stem] = read_image(filename, device=device).unsqueeze(0)

    return input_dict


def save_output_dict(output_dict: Dict[str, th.Tensor], img_folder: PathLike,
                     img_format: str = 'png'):
    """Save a dictionary of images (float tensors) as local files.

    Args:
        output_dict (Dict[str, Tensor]): Source dictionary of image tensors.
        img_folder (PathLike): Output directory path.
        img_format (str, optional): Image file format ('png' or 'exr'). Defaults to 'png'.
    """
    # Verify input correctness
    check_output_dict(output_dict)

    # Create the output folder
    img_folder = Path(img_folder)
    img_folder.mkdir(parents=True, exist_ok=True)

    for filename, img in output_dict.items():
        img = img.detach().squeeze(0) if img.ndim == 4 else img.detach()
        write_image(img, img_folder / filename, img_format=img_format)


def save_output_dict_to_sbs(output_dict: Dict[str, th.Tensor], filename: PathLike):
    """Export a dictionary of output images (SVBRDF maps) to a SBS document. The images are stored
    in a dependency folder and referred to using linked resources.

    Args:
        output_dict (Dict[str, Tensor]): Source dictionary of SVBRDF maps. The keys should match
            `Renderer.CHANNELS` in `diffmat/core/render.py`.
        filename (PathLike): File name of the output SBS document.

    Raises:
        ValueError: The shape of a source image is invalid for exportation, namely, its height or
            width dimension is not an integral power of 2.
    """
    # Determine which SVBRDF maps are saved
    from diffmat.core.material import Renderer
    output_dict = {channel: img for channel, img in output_dict.items() \
                   if channel in Renderer.CHANNELS}
    logger.info(f"The following SVBRDF maps will be saved: {', '.join(list(output_dict.keys()))}")

    # Verify input images are valid to export
    check_output_dict(output_dict)
    valid_shapes = [1 << i for i in range(13)]

    for channel, img in output_dict.items():
        if img.shape[-2] not in valid_shapes or img.shape[-1] not in valid_shapes:
            raise ValueError(f"The shape of image '{channel}' ({list(img.shape)}) is "
                             f"invalid for export")

    # The output size of uniform colors are minimized across all images
    min_size = tuple(min(img.shape[d] for img in output_dict.values()) for d in (-2, -1))
    min_size_log2 = tuple(int(math.log2(i)) for i in min_size)
    min_size_log2_str = ' '.join(str(i) for i in min_size_log2)

    # Create a dependency folder to store the image files
    output_dir = Path(filename).parent
    output_dep_dir = output_dir / 'dependencies'
    output_dep_dir.mkdir(parents=True, exist_ok=True)

    # Save the images to the dependency folder
    save_output_dict(output_dict, output_dep_dir)

    # Read template SBS document files
    template_dir = Path(__file__).parent / 'export_template'
    res_root_et = ET.parse(template_dir / 'export_sbs.sbs').getroot()
    alt_root_et = ET.parse(template_dir / 'export_sbs_const.sbs').getroot()

    # Helper function for replacing the output size parameter of an atomic node
    def set_output_size(node_et: ET.Element, output_size: str):

        # Search for the output size parameter entry
        params = node_et.find('.//compFilter/parameters')
        param_et = params.find("parameter/name[@v='outputsize']/..")

        # Set the value if the parameter entry exists; otherwise, create a new entry
        if param_et is not None:
            param_et.find('relativeTo').set('v', '0')
            param_et.find('paramValue/constantValueInt2').set('v', output_size)
        else:
            param_et = ET.fromstring(
                f'<parameter>'
                f'<name v="outputsize"/><relativeTo v="0"/>'
                f'<paramValue><constantValueInt2 v="{output_size}"/></paramValue>'
                f'</parameter>'
            )
            params.append(param_et)

    # Construct the export SBS document by combining both template files
    for channel in Renderer.CHANNELS.keys():

        # Find the bitmap node that links to the input image
        graph_outputs_et = res_root_et.find('.//graphOutputs')
        graph_output_et = graph_outputs_et.find(f"graphoutput/identifier[@v='{channel}']/..")

        output_uid = graph_output_et.find('uid').get('v')
        comp_nodes_et = res_root_et.find('.//compNodes')
        output_node_et = \
            comp_nodes_et.find(f".//compOutputBridge/output[@v='{output_uid}']/../../..")

        bitmap_uid = output_node_et.find('.//connRef').get('v')
        bitmap_node_et = comp_nodes_et.find(f"compNode/uid[@v='{bitmap_uid}']/..")

        # Replace missing channels by uniform colors
        if channel not in output_dict:

            # Replace the graph output entry related to the missing output map
            graph_outputs_et.remove(graph_output_et)
            graph_output_et = alt_root_et.find(f".//graphoutput/identifier[@v='{channel}']/..")
            graph_outputs_et.append(graph_output_et)

            # Replace node entries related to the missing output map
            comp_nodes_et.remove(output_node_et)
            comp_nodes_et.remove(bitmap_node_et)
            output_node_et = \
                alt_root_et.find(f".//compOutputBridge/output[@v='{output_uid}']/../../..")
            color_uid = output_node_et.find('.//connRef').get('v')
            color_node_et = alt_root_et.find(f".//compNode/uid[@v='{color_uid}']/..")
            comp_nodes_et.append(output_node_et)
            comp_nodes_et.append(color_node_et)

            # Set the output size of the color node
            set_output_size(color_node_et, min_size_log2_str)

            # Drop the linked resource entry
            res_et = res_root_et.find(".//group/identifier[@v='Resources']/../content")
            res_entry_et = res_et.find(f"resource/identifier[@v='{channel}']/..")
            res_et.remove(res_entry_et)

        else:
            # Convert the image size into string
            img = output_dict[channel]
            size_log2 = tuple(int(math.log2(i)) for i in img.shape[-2:])
            size_log2_str = ' '.join(str(i) for i in size_log2)

            # Change the output size of bitmap nodes to match the input image shape
            set_output_size(bitmap_node_et, size_log2_str)

    # Save the output SBS file
    ET.ElementTree(res_root_et).write(filename, encoding='utf-8', xml_declaration=True)
