from pathlib import Path
import argparse
import json
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.nn.functional import interpolate

from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.core.io import read_image
from diffmat.core.util import FILTER_OFF, FILTER_YES
from diffmat.optim import HybridOptimizer
from diffmat.optim.metric import METRIC_DICT


def main():
    """Optimize the integer parameters inside a differentiable procedural material graph to
    match an image-captured material appearance.
    """
    # Default paths
    default_result_dir = Path(__file__).resolve().parents[0] / 'result'

    # Command line argument parser
    prog_description = ('Optimize (all) procedural material graph parameters to match an '
                        'input image')
    parser = argparse.ArgumentParser(description=prog_description)

    ## I/O path
    parser.add_argument('input', metavar='FILE', help='Path to the input *.sbs file')
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

    ## Optimization
    parser.add_argument('-a', '--algorithm', metavar='ALGO', default='grid',
                        help='Algorithm backend for integer optimization')
    parser.add_argument('-n', '--num-iters', type=int, default=5,
                        help='Number of optimization iterations')
    parser.add_argument('-m', '--metric', default='vgg', choices=METRIC_DICT.keys(),
                        help="Texture descriptor type ('vgg', 'fft', or 'combine')")
    parser.add_argument('-mw', '--combine-fft-weight', type=float, default=1e-3,
                        help='FFT term weight in combined loss function')
    parser.add_argument('-s', '--seed', type=int, default=-1, help='Random seed')
    parser.add_argument('-ip', '--init-params', metavar='FILE', default='',
                        help='Specify the initial parameter values using an external file')
    parser.add_argument('-si', '--save-interval', type=int, default=1,
                        help='Number of iterations between two checkpoints')
    parser.add_argument('-lvi', '--filter-integer', type=int, default=FILTER_OFF,
                        help='Integer parameter optimization level')
    parser.add_argument('-lve', '--filter-exposed', type=int, default=FILTER_OFF,
                        help='Exposed continuous parameter optimization level')
    parser.add_argument('-lvg', '--filter-generator', type=int, default=FILTER_YES,
                        help='Generator continuous parameter optimization level')
    parser.add_argument('-lvie', '--filter-int-exposed', type=int, default=FILTER_OFF,
                        help='Exposed integer parameter optimization level')
    parser.add_argument('-lvig', '--filter-int-generator', type=int, default=FILTER_OFF,
                        help='Generator integer parameter optimization level')
    parser.add_argument('-k', '--opt-kwargs', default='{}',
                        help='Other keyword arguments for the optimization function (JSON format)')

    ## Simulated annealing
    parser.add_argument('--t-max', metavar='VAL', type=float, default=5e-3,
                        help='Maximum temperature')
    parser.add_argument('--t-min', metavar='VAL', type=float, default=1e-6,
                        help='Minimum temperature')
    parser.add_argument('--max-steps', metavar='NUM', type=int, default=10000,
                        help='Max number of annealing steps')
    parser.add_argument('--pt-prob', metavar='VAL', type=float, default=0.05,
                        help='Parameter perturbation probability')
    parser.add_argument('--pt-min', metavar='VAL', type=float, default=-0.02,
                        help='Minimal parameter perturbation (fraction of value range)')
    parser.add_argument('--pt-max', metavar='VAL', type=float, default=0.02,
                        help='Minimal parameter perturbation (fraction of value range)')

    ## Grid search
    parser.add_argument('-sr', '--search-res', type=int, default=100,
                        help='Continuous line search resolution in grid search')

    ## Other control
    parser.add_argument('-c', '--cpu', action='store_true', help='Run the test on CPU only')
    parser.add_argument('--exr', action='store_true', help='Load the input in exr format')
    parser.add_argument('--stat-only', action='store_true',
                        help='Only show graph stats and do not run optimization')

    args = parser.parse_args()

    # Configure diffmat logger
    config_logger(args.logging_level)

    # Set up material graph translator and get the translated graph object
    translator = MGT(args.input, args.res, external_noise=args.external_noise,
                     toolkit_path=args.toolkit_path)

    # Create the result folder
    result_dir = Path(args.result_dir) / translator.graph_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subfolder names for external inputs and optimization results
    dir_name = str(args.graph_seed) if args.graph_seed >= 0 else 'default'
    ext_input_dir = result_dir / 'external_input' / dir_name

    output_dir_name = args.output_dir_name or \
                      (f'optim_{Path(args.input_image).stem}' if args.input_image else \
                       'optim_integer')
    output_dir = result_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the translated graph object
    device = 'cpu' if args.cpu else 'cuda'
    graph = translator.translate(
        seed=args.graph_seed, normal_format=args.normal_format,
        external_input_folder=ext_input_dir, device=device)
    graph.compile()

    # Optionally read initial parameter values from an external file
    if args.init_params:
        graph.load_parameters_from_file(args.init_params)

    # Configuration for combined loss function (if used)
    combined_loss_config = {
        'vgg': {'weight': 1.0},
        'fft': {'weight': args.combine_fft_weight},
    }

    # Instantiate the integer parameter optimizer
    optimizer_kwargs = {
        'algorithm': args.algorithm,
        'metric': args.metric,
        'metric_kwargs':
            {'config': combined_loss_config} if args.metric == 'combine' else {},
        'filter_integer': args.filter_integer,
        'filter_exposed': args.filter_exposed,
        'filter_generator': args.filter_generator,
        'filter_int_exposed': args.filter_int_exposed,
        'filter_int_generator': args.filter_int_generator,
    }
    if args.algorithm == 'simanneal':
        optimizer_kwargs['backend_kwargs'] = {
            'seed': args.seed,
            'T_max': args.t_max,
            'T_min': args.t_min,
            'max_steps': args.max_steps,
            'pt_prob': args.pt_prob,
            'pt_min': args.pt_min,
            'pt_max': args.pt_max,
        }

    optimizer = HybridOptimizer(graph, **optimizer_kwargs)

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
    else:
        target_img = graph.evaluate()

    # Run integer optimization
    opt_kwargs = json.loads(args.opt_kwargs)
    if args.algorithm == 'grid':
        opt_kwargs['search_res'] = args.search_res

    optimizer.optimize(target_img, num_iters=args.num_iters, result_dir=output_dir,
                       save_interval=args.save_interval, update_interval=100,
                       save_output_sbs=args.save_output_sbs, img_format=img_format, **opt_kwargs)


if __name__ == '__main__':
    main()
