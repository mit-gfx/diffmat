from pathlib import Path
import json
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import torch as th

from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.core.io import write_image, save_output_dict, save_output_dict_to_sbs
from diffmat.core.material import Renderer


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
    parser.add_argument('-o', '--output-dir-name', metavar='NAME', default='',
                        help='Output folder name in the graph-specific result directory')

    ## Configuration parameters
    parser.add_argument('-x', '--res', type=int, default=9, help='Output image resolution')
    parser.add_argument('-s', '--seed', type=int, default=-1, help='Random seed')
    parser.add_argument('-l', '--logging-level', metavar='LEVEL', default='default',
                        choices=('none', 'quiet', 'default', 'verbose'), help='Logging level')
    parser.add_argument('-t', '--toolkit-path', metavar='PATH', default='',
                        help='Path to Substance Automation Toolkit')
    parser.add_argument('-e', '--external-noise', action='store_true',
                        help='Generating noise textures using SAT only')
    parser.add_argument('-nf', '--normal-format', metavar='FORMAT', default='dx',
                        choices=['dx', 'gl'], help='Output normal format for rendering')
    parser.add_argument('--graph-ablation', action='store_true',
                        help='Generate noise textures using SAT and substitute generator-like '
                             '(FX-Map, Pixel Processor, and other generator) nodes with dummy '
                             'pass-through nodes')

    ## Control switches
    parser.add_argument('-g', '--with-gt', action='store_true',
                        help='Generate ground truth node outputs')
    parser.add_argument('-it', '--iteration', type=int, default=3,
                        help='Number of iterations to repeat for forward and backward evaluations')
    parser.add_argument('-b', '--backward', action='store_true',
                        help='Test gradient backpropagation')
    parser.add_argument('-m', '--save-memory', action='store_true',
                        help='Save all intermediate images (this can be very slow)')
    parser.add_argument('--save-output-sbs', action='store_true',
                        help='Save output images into a SBS document')
    parser.add_argument('--save-annotations', action='store_true',
                        help='Save node-wise numeric annotations into a SBS document')
    parser.add_argument('-c', '--cpu', action='store_true', help='Run the test on CPU only')
    parser.add_argument('--benchmarking', action='store_true',
                        help='Initiate benchmarking in graph evaluation, which times the forward'
                             'and backward pass of every node in the graph')
    parser.add_argument('--debug', action='store_true', help='Gradient debugging switch')

    ## Parameter loading and inspection
    parser.add_argument('-ip', '--init-params', metavar='FILE', default='',
                        help='Specify the initial parameter values using an external file')
    parser.add_argument('-sp', '--save-params', action='store_true',
                        help='Save graph parameters as a JSON configuration file')

    args = parser.parse_args()

    # Configure diffmat logger
    config_logger(args.logging_level)

    # Set gradient debugging
    th.autograd.set_detect_anomaly(args.debug)

    # Set up material graph translator and get the translated graph object
    translator = MGT(
        args.input, args.res, external_noise=args.external_noise, toolkit_path=args.toolkit_path,
        ablation=args.graph_ablation)

    # Create the result folder
    result_dir = Path(args.result_dir) / translator.graph_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subfolder names for external inputs and image outputs
    dir_name = args.output_dir_name if args.output_dir_name else \
               str(args.seed) if args.seed >= 0 else 'default'
    ext_input_dir = result_dir / 'external_input' / dir_name
    output_dir = result_dir / dir_name

    # Translate the graph from XML and save a summary
    device = 'cpu' if args.cpu else 'cuda'
    graph = translator.translate(
        seed=args.seed, normal_format=args.normal_format, external_input_folder=ext_input_dir,
        device=device)
    graph.summarize(result_dir / 'summary.yml')
    graph.compile()

    # Optionally read initial parameter values from an external file
    if args.init_params:
        graph.load_parameters_from_file(args.init_params)

    # Evaluate the graph in forward-only mode
    for _ in range(args.iteration):
        with graph.timer('Forward'):
            img_render = graph.evaluate(benchmarking=args.benchmarking)

    # Test backward pass
    if args.backward > 0:
        print('Test backward:')
        graph.train()
        for _ in range(args.iteration):
            with graph.timer('Forward'):
                img_render = graph.evaluate(benchmarking=args.benchmarking)
            if not args.benchmarking:
                with graph.timer('Backward'):
                    img_render.sum().backward()

    # Save the final image and, optionally, all intermediate outputs to the target folder
    if args.save_memory:
        output_dict = dict((key, val) for key, val in graph.memory.items() if val is not None)
    else:
        output_dict = dict({key: graph.memory[key] for key in Renderer.CHANNELS})

    save_output_dict(output_dict, output_dir)
    write_image(img_render.squeeze(0), output_dir / 'render')

    # Optionally save the current parameter values into a JSON file
    if args.save_params:
        param_config = graph.get_parameters_as_config(constant=True)
        with open(output_dir / 'params.json', 'w') as f:
            json.dump(param_config, f)

    # Optionally save the output image into a SBS document
    export_dir = result_dir / 'export'
    if args.save_output_sbs:
        export_dir.mkdir(parents=True, exist_ok=True)
        save_output_dict_to_sbs(output_dict, export_dir / 'diffmat_output.sbs')

    # Optionally save annotated graph into a SBS document
    if args.save_annotations:
        export_dir.mkdir(parents=True, exist_ok=True)
        translator.export_annotated_graph(export_dir / 'graph_annotations.sbs')

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
