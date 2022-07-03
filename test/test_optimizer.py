from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.nn.functional import interpolate
import torch as th

from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.optim import Optimizer
from diffmat.core.io import read_image


def main():
    """Optimize the continuous parameters inside a differentiable procedural material graph to
    match an image-captured material appearance.
    """
    # Default paths
    default_result_dir = Path(__file__).resolve().parents[0] / 'result'

    # Command line argument parser
    prog_description = 'Optimize procedural material graph parameters to match an input image'
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
    parser.add_argument('--res', type=int, default=9, help='Output image resolution')
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
    parser.add_argument('-n', '--num-iters', type=int, default=1000,
                        help='Number of optimization iterations')
    parser.add_argument('-m', '--metric', default='td', choices=['td', 'fft'],
                        help="Texture descriptor type ('td' or 'fft')")
    parser.add_argument('-ip', '--init-params', metavar='FILE', default='',
                        help='Specify the initial parameter values using an external file')
    parser.add_argument('-lr', '--learning-rate', type=float, default=5e-4,
                        help='Optimization learning rate')
    parser.add_argument('-si', '--save-interval', type=int, default=20,
                        help='Number of iterations between two checkpoints')
    parser.add_argument('-li', '--load-checkpoint-iter', type=int, default=-1,
                        help='Iteration number to load checkpoint from')
    parser.add_argument('-ld', '--load-checkpoint-dir', default='',
                        help="The folder where checkpoints are stored")
    parser.add_argument('-lve', '--opt-level-exposed', type=int, default=2,
                        help='Exposed parameter optimization level')

    ## Other control
    parser.add_argument('-c', '--cpu', action='store_true', help='Run the test on CPU only')
    parser.add_argument('--exr', action='store_true', help='Load the input in exr format')

    args = parser.parse_args()

    # Configure diffmat logger
    config_logger(args.logging_level)

    # Set up material graph translator and get the translated graph object
    translator = MGT(args.input, args.res, external_noise=args.external_noise,
                     toolkit_path=args.toolkit_path)

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
        seed=args.graph_seed, normal_format=args.normal_format, external_input_folder=ext_input_dir,
        device=device)
    graph.compile()

    # Optionally read the initial parameter values from an external file
    if args.init_params:
        init_params: th.Tensor = th.load(args.init_params)['param']
        graph.set_parameters_from_tensor(init_params.to(device))

    # Read the specified input image from local file (e.g., real-world target) and resize it to the
    # target optimization size
    img_format = 'exr' if args.exr else 'png'

    if args.input_image:
        img_size = (1 << args.res, 1 << args.res)
        target_img = read_image(args.input_image)[:3].unsqueeze(0)
        target_img = interpolate(
            target_img, size=img_size, mode='bilinear', align_corners=False)

    # Read a sampled image from local file (synthetic target)
    else:
        input_dir = result_dir / (args.input_dir_name or 'sample_default')
        target_img = read_image(input_dir / 'render' / f'{args.input_file_name}.{img_format}')
        target_img.unsqueeze_(0)

    # Run optimization to match the target image
    optimizer_kwargs = {
        'lr': args.learning_rate,
        'metric': args.metric,
        'opt_level_exposed': args.opt_level_exposed,
    }
    optimizer = Optimizer(graph, **optimizer_kwargs)
    optimizer.optimize(target_img, num_iters = args.num_iters,
                       start_iter = max(args.load_checkpoint_iter, 0),
                       load_checkpoint = args.load_checkpoint_iter >= 0,
                       load_checkpoint_dir = args.load_checkpoint_dir,
                       save_interval = args.save_interval,
                       save_output_sbs = args.save_output_sbs,
                       result_dir = output_dir, img_format = img_format)


if __name__ == '__main__':
    main()
