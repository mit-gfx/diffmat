from xml.etree import ElementTree as ET
from copy import deepcopy
from pathlib import PurePath, Path
from typing import List, Tuple, Dict, Set, Any, Optional, Iterator
import random
import platform
import subprocess
import re
import json
import logging

import torch as th

from diffmat.core.io import load_input_dict
from diffmat.translator import types as tp
from diffmat.translator.node_trans import ExternalInputNT
from diffmat.translator.types import PathLike, DeviceType
from diffmat.translator.util import has_connections, to_constant, get_value


class ExtInputGenerator:
    """Generator for external noises or other input images using Substance Automation Toolkit.
    """
    def __init__(self, root: ET.Element, res: int, toolkit_path: Optional[PathLike] = None):
        """Initialize the external input generator by an example XML tree.

        Args:
            root (Element): Root node of the XML tree that encodes the material graph.
            res (int): Output texture resolution.
            toolkit_path (Optional[PathLike], optional): Path to the executables of Substance
                Automation Toolkit. Passing None prompts the generator to use a OS-specific
                default location:
                    Linux/Mac: the $HOME directory.
                    Windows: the Desktop directory.
                Defaults to None.

        Raises:
            RuntimeError: Toolkit path must be specified for an unknown OS platform.
            FileNotFoundError: Substance Automation Toolkit is not found in the provided folder.
            RuntimeError: Failed to read Substance Automation Toolkit version.
        """
        self.source_root = root
        self.res = res

        # The ground-truth tree copy is used to generate ground truth outputs
        self.gt_root = deepcopy(self.source_root)

        # Register a logger for the external input generator
        self.logger = logging.getLogger('diffmat.external')

        # Get the name of the operating system
        self.system_name = platform.system()

        # Use the provided toolkit path or assume an OS-specific default path
        if not toolkit_path:

            # Infer the Substance Designer installation directory
            sd_install_path_os = {
                'Linux': '/opt/Adobe/Adobe_Substance_3D_Designer',
                'Darwin': '/Applications/Adobe Substance 3D Designer/Adobe Substance 3D Designer.'
                          'app/Contents/MacOS',
                'Windows': 'c:\\Program Files\\Adobe\\Adobe Substance 3D Designer'
            }

            if self.system_name in sd_install_path_os:
                toolkit_path = sd_install_path_os[self.system_name]
            else:
                raise RuntimeError(f'Please specify a toolkit path for the unknown platform: '
                                   f'{self.system_name}')

        self.toolkit_path = PurePath(toolkit_path)

        # Detect SAT version
        cooker_name = 'sbscooker.exe' if self.system_name == 'Windows' else 'sbscooker'
        cooker_path = Path(self.toolkit_path) / cooker_name
        if not cooker_path.exists():
            raise FileNotFoundError("Substance Automation Toolkit is not found. Please provide "
                                    "the path to the folder where 'sbscooker' and 'sbsrender' "
                                    "executables are located.")

        output = subprocess.run(f'"{cooker_path}" --version', shell=True, capture_output=True,
                                text=True)
        version: Optional[str] = re.search(r'(?<= )\d{2,4}\.\d\.\d(?= )', output.stdout)[0]
        if version is None:
            raise RuntimeError('Failed to obtain Substance Automation Tookit version')

        self.toolkit_version = version

        # Reserved fields for automatically fixing SAT-related errors
        self.toolkit_version_fix = ''
        self.cpu_engine_fix = ''

        # Copy and clean up the source XML tree as a template
        self.reset()

    def reset(self):
        """Purge the I/O material graph of nodes and outputs, only retaining a template of the
        source material graph. Meanwhile, prepare the I/O material graph and the ground truth graph
        copy by resetting their output resolutions and version info.
        """
        # Initialize I/O material graph using a template of the source material graph
        root = deepcopy(self.source_root)
        root.find('dependencies').clear()
        root.find('.//graphOutputs').clear()
        root.find('.//compNodes').clear()
        root.find('.//root/rootOutputs').clear()
        self.root = root

        for r in (self.root, self.gt_root):

            # Retain self dependency
            dep_self_et = self.source_root.find(
                "dependencies/dependency/filename[@v='?himself']/..")
            if r is self.root and dep_self_et is not None:
                r.find('dependencies').append(dep_self_et)

            # Set output resolution
            param_et = r.find(".//baseParameters/parameter/name[@v='outputsize']/..")
            if param_et is not None:
                param_et.find('relativeTo').set('v', '0')
                param_et.find('paramValue/constantValueInt2').set('v', f'{self.res} {self.res}')
            else:
                param_et = ET.fromstring(
                    f'<parameter>'
                    f'  <name v="outputsize"/>'
                    f'  <relativeTo v="0"/>'
                    f'  <paramValue><constantValueInt2 v="{self.res} {self.res}"/></paramValue>'
                    f'</parameter>'
                )
                r.find('.//baseParameters').append(param_et)

            # Apply the SAT version fix if any
            if self.toolkit_version_fix:
                r.find('formatVersion').set('v', self.toolkit_version_fix)
                r.find('updaterVersion').set('v', self.toolkit_version_fix)

    def _add_source_node(self, node_et: ET.Element, node_gpos: Tuple[int, int, int],
                         reset_random_seed: bool = False):
        """Add a modified copy of the source generator node to the I/O material graph.

        Args:
            node_et (Element): Root node of the XML subtree for a material graph node.
            node_gpos (Tuple[int, int, int]): Node position in Substance Designer GUI.
            reset_random_seed (bool, optional): Switch for clearing the random seed parameter of
                source material nodes. This is useful when generating textures using a unified
                random seed. Defaults to False.
        """
        node_fix_et = deepcopy(node_et)
        node_params_et = node_fix_et.find('.//parameters')

        # Helper function that removes parameters that match a given name
        def remove_param(name: str, keep_dynamic: bool = True) -> bool:

            # Succeed if the parameter has been removed (or simply doesn't exist)
            param_et = node_params_et.find(f"parameter/name[@v='{name}']/..")
            if param_et is None:
                return True

            # Determine whether the parameter should be removed
            # 'randomseed' parameters are always removed for generator nodes
            is_dynamic = param_et.find('.//dynamicValue') is not None
            to_remove = name == 'randomseed' and not has_connections(node_et) \
                        or not is_dynamic or not keep_dynamic

            # Warn the user if the parameter has a function graph which might affect output
            if is_dynamic and to_remove:
                self.logger.warn(f"A '{name}' parameter to be removed is defined by a function"
                                 f" graph. The generated outputs might be incorrect.")

            # Remove the parameter
            if to_remove:
                node_params_et.remove(param_et)

            return to_remove

        # Enforce the output size to inherit from the global value
        remove_param('outputsize')

        # Add or reset its 'randomseed' parameter for later use
        if reset_random_seed and remove_param('randomseed'):
            node_params_et.append(ET.fromstring(
                '<parameter>'
                '  <name v="randomseed"/>'
                '  <relativeTo v="1"/>'
                '  <paramValue><constantValueInt32 v="0"/></paramValue>'
                '</parameter>'
            ))

        # Set the position of the modified source node in the I/O material graph
        x, y, z = node_gpos
        node_fix_et.find('GUILayout/gpos').set('v', f'{x} {y} {z}')

        # Finally, attach the node to the I/O material graph
        self.root.find('.//compNodes').append(node_fix_et)


    def _add_output_node(self, node_name: str, node_uid: int, node_gpos: Tuple[int, int, int],
                         output_name: str, output_uid: int, output_index: int = 0,
                         pos_stride: int = 100, output_suffix: str = ''):
        """Generate an output node for a source generator node, and add the output node to the I/O
        material graph.

        Args:
            node_name (str): Source material node name.
            node_uid (int): Source material node UID.
            node_gpos (Tuple[int, int, int]): Material node position in Substance Designer GUI.
            output_name (str): Output node name.
            output_uid (int): Output node UID.
            output_index (int, optional): Index of the output node if multiple output nodes must be
                generated for the source material node. Defaults to 0.
            pos_stride (int, optional): Position stride in Substance Designer GUI. Defaults to 100.
            output_suffix (str, optional): Optionally append a suffix to the output node name.
                Defaults to ''.
        """
        # Compute output node UID and global output channel UID by applying a random offset to node
        # and output slot UIDs
        rand_l, rand_r = int(5e8), int(6e8)
        output_node_uid = node_uid + random.randint(rand_l, rand_r)
        output_channel_uid = output_uid + random.randint(rand_l, rand_r)

        # Calculate the output node position
        x, y, z = node_gpos
        output_node_gpos = f'{x + (output_index + 2) * pos_stride} {y} {z}'

        # Add the output node info to the I/O material graph
        output_node_et = ET.fromstring(
            f'<compNode>'
            f'  <uid v="{output_node_uid}"/>'
            f'  <GUILayout><gpos v="{output_node_gpos}"/></GUILayout>'
            f'  <connections><connection>'
            f'    <identifier v="inputNodeOutput"/>'
            f'    <connRef v="{node_uid}"/>'
            f'    <connRefOutput v="{output_uid}"/>'
            f'  </connection></connections>'
            f'  <compImplementation><compOutputBridge>'
            f'    <output v="{output_channel_uid}"/>'
            f'  </compOutputBridge></compImplementation>'
            f'</compNode>'
        )
        self.root.find('.//compNodes').append(output_node_et)

        # Name the output channel
        output_channel_name = (
            f"{node_name}{'_' if output_name else ''}"
            f"{output_name}{output_suffix}"
        )

        # Add a graph output entry
        graph_output_et = ET.fromstring(
            f'<graphoutput>'
            f'  <identifier v="{output_channel_name}"/>'
            f'  <uid v="{output_channel_uid}"/>'
            f'  <channels v="2"/><group v="input"/>'
            f'</graphoutput>'
        )
        self.root.find('.//graphOutputs').append(graph_output_et)

        # Add a global output entry
        root_output_et = ET.fromstring(
            f'<rootOutput>'
            f'  <output v="{output_channel_uid}"/>'
            f'  <format v="0"/><usertag v=""/>'
            f'</rootOutput>'
        )
        self.root.find('.//rootOutputs').append(root_output_et)

    def _create_output_nodes(self, node_trans: List[ExternalInputNT], node_suffix: str = '',
                             reset_random_seed: bool = False):
        """Add a collection of output nodes to the new I/O material graph, which correspond to
        output channels of the source nodes.

        Args:
            node_trans (List[ExternalInputNT]): List of external input node translators that
                interface with source noise/pattern generator nodes.
            node_suffix (str, optional): Optionally append a common suffix to the names of the
                created output nodes. Defaults to ''.
            reset_random_seed (bool, optional): Switch for clearing the random seed parameter of
                source material nodes. This is useful when generating textures using a unified
                random seed. Defaults to False.
        """
        # A reference position for locating subsequent nodes
        ref_pos: Tuple[int, int, int] = tuple()

        # The spacing between adjacent nodes in the I/O material graph
        node_pos_stride: int = 128

        # Look-up table of dependency information by UID
        dep_dict: Dict[int, ET.Element] = \
            {int(n.find('uid').get('v')): n for n in self.source_root.iter('dependency')}
        added_deps: Set[int] = set()

        # Helper iterator that enumerates connected output slots for a source node
        def non_empty_output(trans: ExternalInputNT) -> Iterator[str]:
            return (key for key, val in trans.outputs.items() if val)

        # Helper function that looks up an output connector name in Substance designer given the
        # diffmat name
        def reverse_lookup(node_out_name: str) -> str:
            return next(key for key, val in trans.node_config['output'].items() \
                        if val == node_out_name)

        # Generating records in the XML template for each generator node
        for order, trans in enumerate(node_trans):

            # Obtain node name
            node_name = trans.name

            # Obtain node UID, implementation tag, and position coordinates
            node_et: ET.Element = trans.root
            node_uid = int(node_et.find('uid').get('v'))
            node_imp_et = node_et.find('compImplementation')[0]
            node_tag = node_imp_et.tag
            node_gpos: List[int] = to_constant(node_et.find('GUILayout/gpos').get('v'), tp.FLOAT3)

            # Set the position of the source node in the I/O material graph
            # Initialize the reference position if necessary
            if ref_pos:
                x, y, z = ref_pos
                gpos = (x, y + order * node_pos_stride, z)
            else:
                ref_pos = tuple(node_gpos)
                gpos = ref_pos

            # Insert a modified copy of the source node to the material I/O material graph
            self._add_source_node(node_et, gpos, reset_random_seed=reset_random_seed)

            # Non-atomic generator nodes (which might have multiple outputs)
            if node_tag == 'compInstance':

                # Obtain dependency UID and insert the corresponding dependency information into
                # the new XML tree
                dep_uid = int(get_value(node_imp_et.find('path')).split('=')[1])
                if dep_uid not in added_deps:
                    added_deps.add(dep_uid)
                    self.root.find('dependencies').append(dep_dict[dep_uid])

                # Add a global output node for each output connector slot with connections
                for i, node_out_name in enumerate(non_empty_output(trans)):

                    # Reverse look-up the output identifier in Substance designer
                    node_out_name_sbs = reverse_lookup(node_out_name)
                    node_out_et = node_imp_et.find(
                        f"outputBridgings/outputBridging/identifier[@v='{node_out_name_sbs}']/..")
                    if node_out_et is None:
                        self.logger.critical(
                            f"Output '{node_out_name}' of node '{node_name}' is not found")

                    node_out_uid = int(node_out_et.find('uid').get('v'))
                    self._add_output_node(
                        node_name, node_uid, gpos, node_out_name, node_out_uid, output_index=i,
                        pos_stride=node_pos_stride, output_suffix=node_suffix)

            # Atomic generator nodes (one output only)
            else:
                node_out_et = node_et.find('compOutputs/compOutput')
                node_out_uid = int(node_out_et.find('uid').get('v'))
                self._add_output_node(
                    node_name, node_uid, gpos, '', node_out_uid, pos_stride=node_pos_stride,
                    output_suffix=node_suffix)

    def _run_sat_command(self, command: str) -> str:
        """Executes an SAT command and detect warnings/errors reported by the program.

        Args:
            command (str): The shell command to execute.

        Raises:
            DeprecationWarning: The SAT version is too old for the *.sbs document.
            RuntimeError: Substance Automation Toolkit command failed, error messages returned.

        Returns:
            str: Standard output of the SAT command.
        """
        # Read from standard output of the SAT command
        output = subprocess.run(command, shell=True, capture_output=True, text=True)
        stderr, stdout = output.stderr, output.stdout

        # Detect and handle errors
        if '[ERROR]' in stderr:

            # Version compatilibity issue
            if 'Application is too old' in stderr:
                self.logger.warn(
                    f"The SAT version '{self.toolkit_version}' appears to be too old for the "
                    f"*.sbs document. Attempting to fix with backward compatibility..."
                )
                raise DeprecationWarning(stderr)

            # Unknown error type
            else:
                self.logger.critical(f'Error messages from SAT:\n{stderr}')
                raise RuntimeError('SAT command execution has failed. Please see the error'
                                   'messages above')

        # Unknown warnings
        elif '[WARNING]' in stderr:
            self.logger.warn(f'Warning messages from SAT:\n{stderr}')

        return stdout

    # Executes an sbscooker command
    _run_sbscooker_command = _run_sat_command

    def _run_sbsrender_command(self, command: str) -> str:
        """Executes an sbsrender command and parse the output to make sure that the texture images
        are properly generated.

        Args:
            command (str): The shell command to execute.

        Raises:
            RuntimeWarning: The default rendering engine of 'sbsrender' fails to generate output
                textures.

        Returns:
            str: Standard output of the sbsrender command.
        """
        # Execute the rendering command and retrieve the output
        output = self._run_sat_command(command)

        # Parse the output in JSON
        output_json: List[Dict[str, List[Dict[str, Any]]]] = json.loads(output)

        # Detect output textures
        if not output_json[0]['outputs']:
            self.logger.warning(f"'sbsrender' failed to generate output textures using the "
                                 "default rendering engine. Switching to CPU (SSE2) ...")
            raise RuntimeWarning()

        return output

    def _generate_textures(self, seed: int, img_format: str, mode: str = 'input',
                           file_name: str = '', result_folder: PathLike = '.'):
        """Generate input texture maps using Substance Automation Toolkit.

        Args:
            seed (int): Specify a unified random seed for all source material nodes. When the seed
                is negative, the original random seeds of source material nodes will be used.
            img_format (str): Texture image format.
            mode (str, optional): Texture generation mode ('input' or 'output').
                'input': Generating input textures, including noises and patterns.
                'output': Generating ground-truth output SVBRDF maps.
                Defaults to 'input'.
            file_name (str, optional): SBS document file name for the saved I/O material graph.
                Defaults to ''.
            result_folder (PathLike, optional): Output directory for generated textures.
                Defaults to '.'.

        Raises:
            ValueError: Unknown texture generation mode.
        """
        if mode not in ('input', 'output'):
            raise ValueError(f'Unrecognized texture generation mode: {mode}')

        # Create the result folder
        result_folder.mkdir(parents=True, exist_ok=True)

        # Substance document related file names
        sbs_file_name = (result_folder / file_name).with_suffix('.sbs')
        sbsar_file_name = (result_folder / file_name).with_suffix('.sbsar')

        # Toolkit executable names
        cooker_path = self.toolkit_path / 'sbscooker'
        render_path = self.toolkit_path / 'sbsrender'

        # Assemble cooker, render, and cleanup commands
        command_cooker = (
            f'"{cooker_path}" "{sbs_file_name}" --output-path "{result_folder}"'
        )
        command_render = (
            f'"{render_path}" render "{sbsar_file_name}" --output-format "{img_format}" '
            f'--output-name "{{outputNodeName}}" --output-path "{result_folder}"'
            f'{self.cpu_engine_fix}'
        )

        # Build an element tree for writing XML
        root = self.root if mode == 'input' else self.gt_root
        tree = ET.ElementTree(root)

        # Set the random seed for all generator nodes
        if seed >= 0:
            for node_et in (n for n in root.iter('compNode') if not has_connections(n)):
                node_imp_et = node_et.find('compImplementation')[0]
                param_et = node_imp_et.find("parameters/parameter/name[@v='randomseed']/..")
                param_et.find('paramValue/constantValueInt32').set('v', str(seed))

        # Run cooker and render in a loop to fix potential errors
        while True:
            try:
                # Render generator textures
                tree.write(sbs_file_name, encoding='utf-8', xml_declaration=True)
                self._run_sbscooker_command(command_cooker)
                self._run_sbsrender_command(command_render)

            # Handle version compatibility issue
            except DeprecationWarning as e:

                # Replace the version info in the source XML document by the latest supported
                # version and try again
                version_str = re.search(r'(?<=")[\d\.]+(?=")', e.args[0])[0]
                for r in (self.root, self.gt_root):
                    r.find('formatVersion').set('v', version_str)
                    r.find('updaterVersion').set('v', version_str)
                self.toolkit_version_fix = version_str

            # Handle empty texture output from 'sbsrender' by switching to CPU rendering
            except RuntimeWarning:
                self.cpu_engine_fix = ' --engine sse2'
                command_render += self.cpu_engine_fix

            # Exit the loop if no error is detected
            else:
                break

        # Delete the *.sbsar file
        rm = 'del' if self.system_name == 'Windows' else 'rm'
        command_del = f'{rm} "{sbsar_file_name}"'
        subprocess.run(command_del, shell=True, capture_output=True)

    def process(self, node_trans: List[ExternalInputNT], seed: int = -1, img_format: str = 'png',
                suffix: str = '', result_folder: PathLike = '.', read_output_images: bool = True,
                device: DeviceType = 'cpu') -> Optional[Dict[str, th.Tensor]]:
        """Process and generate external inputs based on their XML records.

        Args:
            node_trans (List[ExternalInputNT]): List of external input node translators that
                interface with source noise/pattern generator nodes.
            seed (int, optional): Specify a unified random seed for all source material nodes. When
                the seed is negative, the original random seeds of source material nodes will be
                used. Defaults to -1.
            img_format (str, optional): Output texture image format. Defaults to 'png'.
            suffix (str, optional): An additional suffix to the names of output images.
                Defaults to ''.
            result_folder (PathLike, optional): Output directory for generated textures.
                Defaults to '.'.
            read_output_images (bool, optional): When set to True, the generated textures are read
                by Diffmat into a texture dictionary. Defaults to True.
            device (DeviceType, optional): Device placement of image tensors in the aforementioned
                texture dictionary. Defaults to 'cpu'.

        Returns:
            Optional[Dict[str, Tensor]]: Dictionary of texture maps read from generated images.
        """
        # Exit if the input is empty, i.e., no node to process
        if not node_trans:
            return {}

        # Save the random number generator state to avoid disturbing other random number generation
        # behaviors elsewhere in the system
        rng_state = random.getstate()
        random.seed(seed)

        # Generate companion output nodes for all source nodes.
        self._create_output_nodes(node_trans, node_suffix=suffix, reset_random_seed=seed>=0)

        # Call Substance Automation Toolkit to generate the textures
        graph_name = self.source_root.find('content/graph/identifier').get('v')
        self._generate_textures(
            seed, img_format, file_name=f'{graph_name}_input{suffix}', result_folder=result_folder)

        # Restore random number generator state
        random.setstate(rng_state)

        # Load generated textures from the result folder
        if read_output_images:
            external_inputs = load_input_dict(
                result_folder, glob_pattern=f'*.{img_format}', device=device)

            return external_inputs

    def process_gt(self, seed: int = -1, img_format: str = 'png', suffix: str = '',
                   result_folder: PathLike = '.'):
        """Generate ground-truth texture maps using SAT.

        Args:
            seed (int, optional): Specify a unified random seed for all source material nodes. When
                the seed is negative, the original random seeds of source material nodes will be
                used. Defaults to -1.
            img_format (str, optional): Output texture image format. Defaults to 'png'.
            suffix (str, optional): An additional suffix to the names of output images.
                Defaults to ''.
            result_folder (PathLike, optional): Output directory for generated textures.
                Defaults to '.'.
        """
        # Change the output name of the ground truth graph
        for output_et in self.gt_root.iterfind('.//graphOutputs/graphoutput/identifier'):
            output_et.set('v', f"{output_et.get('v')}{suffix}")

        # Call Substance Automation Toolkit to generate the textures
        graph_name = self.source_root.find('content/graph/identifier').get('v')
        self._generate_textures(
            seed, img_format, mode='output', file_name=f'{graph_name}{suffix}',
            result_folder=result_folder)
