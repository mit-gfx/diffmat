from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.nn.functional import interpolate
import torch as th

from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.optim import Optimizer
from diffmat.optim.metric import METRIC_DICT
from diffmat.core.io import read_image
from diffmat.core.util import FILTER_OFF


def main():
    """Optimize the continuous parameters inside a differentiable procedural material graph to
    match an image-captured material appearance.
    """
    # Default paths
    default_result_dir = Path(__file__).resolve().parents[0] / 'result'

    # Command line argument parser
    prog_description = ('Optimize (continuous) procedural material graph parameters to match an '
                        'input image')
    parser = argparse.ArgumentParser(description=prog_description)

    ## I/O path
    parser.add_argument('input', metavar='FILE', help='Path to the input *.sbs file')
    parser.add_argument('-i', '--input-dir-name', metavar='PATH', default='',
                        help='Input folder path that contains image samples')
    parser.add_argument('-f', '--input-file-name', metavar='NAME', default='params_0',
                        help='File name of input images (without suffix)')
    parser.add_argument('-r', '--result-dir', metavar='PATH', default=str(default_result_dir),
                        help='Result folder path (a separate subfolder is created for each graph')
    parser.add_argument('-o', '--output-dir-name', metavar='NAME', default='',
                        help='Output folder name in the graph-specific result directory')
    parser.add_argument('-im', '--input-image', metavar='FILE', default='',
                        help='Specify an input image (e.g., a real photograph) in the place of '
                             'randomly sampled textures')

    ## Graph related
    parser.add_argument('-x', '--res', type=int, default=9, help='Output image resolution')
    parser.add_argument('-l', '--logging-level', metavar='LEVEL', default='default',
                        choices=('none', 'quiet', 'default', 'verbose'), help='Logging level')
    parser.add_argument('-t', '--toolkit-path', metavar='PATH', default='',
                        help='Path to Substance Automation Toolkit')
    parser.add_argument('-e', '--external-noise', action='store_true',
                        help='Generate noise textures using SAT only')
    parser.add_argument('-nf', '--normal-format', metavar='FORMAT', default='dx',
                        choices=['dx', 'gl'], help='Output normal format for rendering')
    parser.add_argument('-gs', '--graph-seed', type=int, default=-1,
                        help='Material graph master seed')
    parser.add_argument('--save-output-sbs', action='store_true',
                        help='Save the optimized images in an SBS document')
    parser.add_argument('--graph-ablation', action='store_true',
                        help='Generate noise textures using SAT and substitute generator-like '
                             '(FX-Map, Pixel Processor, and other generator) nodes with dummy '
                             'pass-through nodes')

    ## Optimization
    parser.add_argument('-n', '--num-iters', type=int, default=2000,
                        help='Number of optimization iterations')
    parser.add_argument('-m', '--metric', default='vgg', choices=METRIC_DICT.keys(),
                        help="Texture descriptor type ('vgg' or 'fft')")
    parser.add_argument('-ip', '--init-params', metavar='FILE', default='',
                        help='Specify the initial parameter values using an external file')
    parser.add_argument('-lr', '--learning-rate', type=float, default=5e-4,
                        help='Optimization learning rate')
    parser.add_argument('-si', '--save-interval', type=int, default=100,
                        help='Number of iterations between two checkpoints')
    parser.add_argument('-ld', '--load-checkpoint-file', default='',
                        help="The checkpoint file to load into the optimizer")
    parser.add_argument('-lve', '--filter-exposed', type=int, default=FILTER_OFF,
                        help='Exposed parameter optimization level')
    parser.add_argument('-lvg', '--filter-generator', type=int, default=FILTER_OFF,
                        help='Generator parameter optimization level')
    parser.add_argument('-ab', '--ablation-mode', default='none',
                        choices=['none', 'node', 'subgraph'],
                        help='Exclude generator-like nodes from optimization')

    ## Other control
    parser.add_argument('-c', '--cpu', action='store_true', help='Run the test on CPU only')
    parser.add_argument('--exr', action='store_true', help='Load the input in exr format')
    parser.add_argument('--stat-only', action='store_true',
                        help='Only show graph stats and do not run optimization')
    parser.add_argument('--debug', action='store_true', help='Activate gradient debug mode')

    args = parser.parse_args()

    # Activate gradient debugging mode
    th.autograd.set_detect_anomaly(args.debug)

    # Configure diffmat logger
    config_logger(args.logging_level)

    # Set up material graph translator and get the translated graph object
    translator = MGT(args.input, args.res, external_noise=args.external_noise,
                     toolkit_path=args.toolkit_path, ablation=args.graph_ablation)

    # Create the result folder
    result_dir = Path(args.result_dir) / translator.graph_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subfolder names for external inputs and sampling results
    dir_name = str(args.graph_seed) if args.graph_seed >= 0 else 'default'
    ext_input_dir = result_dir / 'external_input' / dir_name

    output_dir_name = args.output_dir_name or \
                      (f'optim_{Path(args.input_image).stem}' if args.input_image else \
                       f'optim_{args.input_file_name}')
    output_dir = result_dir / output_dir_name

    # Get the translated graph object
    device = 'cpu' if args.cpu else 'cuda'
    graph = translator.translate(
        seed=args.graph_seed, normal_format=args.normal_format,
        external_input_folder=ext_input_dir, device=device)
    graph.compile()

    # Optionally read initial parameter values from an external file
    if args.init_params:
        graph.load_parameters_from_file(args.init_params)

    # Construct the optimizer
    optimizer_kwargs = {
        'lr': args.learning_rate,
        'metric': args.metric,
        'filter_exposed': args.filter_exposed,
        'filter_generator': args.filter_generator,
        'ablation_mode': args.ablation_mode,
    }
    optimizer = Optimizer(graph, **optimizer_kwargs)

    # Exit if the user only wants to show graph statistics
    if args.stat_only:
        quit()

    # Read the specified input image from local file (e.g., real-world target) and resize it to the
    # target optimization size
    img_format = 'exr' if args.exr else 'png'

    if args.input_image:
        img_size = (1 << args.res, 1 << args.res)
        target_img = read_image(args.input_image, device=device)[:3].unsqueeze(0)
        target_img = interpolate(target_img, size=img_size, mode='bilinear', align_corners=False)

    # Read a sampled image from local file (synthetic target)
    else:
        input_dir = result_dir / (args.input_dir_name or 'sample_default')
        img_file = input_dir / 'render' / f'{args.input_file_name}.{img_format}'
        target_img = read_image(img_file, device=device).unsqueeze(0)

    # Run optimization to match the target image
    run_kwargs = {
        'num_iters': args.num_iters,
        'result_dir': output_dir,
        'load_checkpoint_file': args.load_checkpoint_file,
        'save_interval': args.save_interval,
        'save_output_sbs': args.save_output_sbs,
        'img_format': img_format,
    }
    optimizer.optimize(target_img, **run_kwargs)


if __name__ == '__main__':
    main()
