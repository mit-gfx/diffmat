from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import torch as th

from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.core.io import write_image, save_output_dict, save_output_dict_to_sbs
from diffmat.core.render import Renderer


def main():
    """Compare diffmat's output with Substance Designer using a chain-like graph.
    """
    # Default paths
    default_result_dir = Path(__file__).resolve().parents[0] / 'result'

    # Command line argument parser
    parser = argparse.ArgumentParser(description='Validate diffmat outputs against Substance.')

    ## I/O
    parser.add_argument('input', metavar='FILE', help='Path to the input *.sbs file')
    parser.add_argument('-r', '--result-dir', metavar='PATH', default=str(default_result_dir),
                        help='Result folder path (a separate subfolder is created for each graph')

    ## Configuration parameters
    parser.add_argument('--res', type=int, default=9, help='Output image resolution')
    parser.add_argument('-s', '--seed', type=int, default=-1, help='Random seed')
    parser.add_argument('-t', '--toolkit-path', metavar='PATH', default='',
                        help='Path to Substance Automation Toolkit')
    parser.add_argument('-nf', '--normal-format', metavar='FORMAT', default='dx',
                        choices=['dx', 'gl'], help='Output normal format for rendering')

    ## Control switches
    parser.add_argument('-e', '--external-noise', action='store_true',
                        help='Generating noise textures using SAT only')
    parser.add_argument('-g', '--with-gt', action='store_true',
                        help='Generate ground truth node outputs')
    parser.add_argument('-b', '--backward', action='store_true',
                        help='Test gradient backpropagation')
    parser.add_argument('-m', '--save-memory', action='store_true',
                        help='Save all intermediate images (this can be very slow)')
    parser.add_argument('--save-output-sbs', action='store_true',
                        help='Save output images into a SBS document')
    parser.add_argument('-c', '--cpu', action='store_true', help='Run the test on CPU only')

    ## On-screen logging
    parser.add_argument('-l', '--logging-level', metavar='LEVEL', default='default',
                        choices=('none', 'quiet', 'default', 'verbose'), help='Logging level')

    args = parser.parse_args()

    # Configure diffmat logger
    config_logger(args.logging_level)

    # Set up material graph translator and get the translated graph object
    translator = MGT(
        args.input, args.res, external_noise=args.external_noise, toolkit_path=args.toolkit_path)

    # Create the result folder
    result_dir = Path(args.result_dir) / translator.graph_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subfolder names for external inputs and image outputs
    dir_name = str(args.seed) if args.seed >= 0 else 'default'
    ext_input_dir = result_dir / 'external_input' / dir_name
    output_dir = result_dir / dir_name

    # Translate the graph from XML and save a summary
    device = 'cpu' if args.cpu else 'cuda'
    graph = translator.translate(
        seed=args.seed, normal_format=args.normal_format, external_input_folder=ext_input_dir,
        device=device)
    graph.summarize(result_dir / 'summary.yml')

    # Compile and evaluate the graph
    graph.compile()
    img_render = graph.evaluate()

    # Test backward pass
    if args.backward:
        print('Test backward:')
        graph.train()
        img_render = graph.evaluate()
        img_render.sum().backward()

    # Save the final image and, optionally, all intermediate outputs to the target folder
    if args.save_memory:
        output_dict = dict((key, val) for key, val in graph.memory.items() if val is not None)
    else:
        output_dict = dict({key: graph.memory[key] for key in Renderer.CHANNELS})

    save_output_dict(output_dict, output_dir)
    write_image(img_render.squeeze(0), output_dir / 'render')

    # Optionally save the output image into a SBS document
    if args.save_output_sbs:
        save_output_dict_to_sbs(output_dict, result_dir / 'export' / 'diffmat_output.sbs')

    # Evaluate the graph using SAT and save the ground truth results
    if args.with_gt:

        # Extract and reset the external input generator
        gt_generator = translator.input_generator
        gt_generator.reset()

        # Save the SAT texture map results
        gt_generator.process_gt(seed=args.seed, suffix='_gt', result_folder=output_dir)

        if args.save_memory:

            # Compile a list of node translators in topological order
            trans_list = (t for t in translator.node_translators \
                          if t.type not in ('input', 'output'))
            trans_dict = {t.name: t for t in trans_list}
            trans_list = [trans_dict[n.name] for n in graph.nodes if n.name in trans_dict]
            gt_generator.process(
                trans_list, seed=args.seed, suffix='_gt', result_folder=output_dir,
                read_output_images=False)


if __name__ == '__main__':
    # TF32 switch on Pytorch
    th.backends.cudnn.allow_tf32 = True
    th.backends.cudnn.enabled = True

    main()
