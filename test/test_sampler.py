from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse

from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.optim import ParamSampler


def main():
    """Randomly sample the continuous parameters in a differentiable procedural material graph.
    """
    # Default paths
    test_root_dir = Path(__file__).resolve().parents[0]
    default_result_dir = test_root_dir / 'result'

    # Command line argument parser
    parser = argparse.ArgumentParser(description='Sample procedural material graph parameters.')

    ## I/O path
    parser.add_argument('input', metavar='FILE', help='Path to the input *.sbs file')
    parser.add_argument('-r', '--result-dir', metavar='PATH', default=str(default_result_dir),
                        help='Result folder path (a separate subfolder is created for each graph')
    parser.add_argument('-o', '--output-dir-name', metavar='NAME', default='',
                        help='Output folder name in the result directory')
    parser.add_argument('-f', '--config-file', metavar='FILE', default='',
                        help='Sampling configuration file')

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

    ## Random parameter sampling
    parser.add_argument('-a', '--sampling-algo', default='uniform',
                        help='Random parameter sampling algorithm')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random parameter sampling seed')
    parser.add_argument('-ns', '--num-samples', metavar='NUM', type=int, default=1,
                        help='Number of random parameter sets')
    parser.add_argument('-sve', '--sampling-level-exposed', type=int, default=2,
                        help='Exposed parameter sampling level')

    ## Other control
    parser.add_argument('-c', '--cpu', action='store_true', help='Run the test on CPU only')
    parser.add_argument('--save-exr', action='store_true', help='Save files as exr format')

    args, algo_args = parser.parse_known_args()

    # Parse algorithm-related arguments
    algo_parser = argparse.ArgumentParser()
    algo_parser.add_argument('--min', type=float, default=-0.05)
    algo_parser.add_argument('--max', type=float, default=0.05)
    algo_parser.add_argument('--mu', type=float, default=0.0)
    algo_parser.add_argument('--sigma', type=float, default=0.03)

    algo_kwargs = vars(algo_parser.parse_args(algo_args))

    # Configure diffmat logger
    config_logger(args.logging_level)

    # Set up the material graph translator
    translator = MGT(args.input, args.res, external_noise=args.external_noise,
                     toolkit_path=args.toolkit_path)

    # Create the result folder
    result_dir = Path(args.result_dir) / translator.graph_name
    result_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subfolder names for external inputs and sampling results
    dir_name = str(args.graph_seed) if args.graph_seed >= 0 else 'default'
    ext_input_dir = result_dir / 'external_input' / dir_name
    output_dir = result_dir / (args.output_dir_name or f'sample_{dir_name}')

    # Get the translated graph object
    device = 'cpu' if args.cpu else 'cuda'
    graph = translator.translate(
        seed=args.graph_seed, normal_format=args.normal_format, external_input_folder=ext_input_dir,
        device=device)
    graph.compile()

    # Run random sampler
    sampler = ParamSampler(graph, algo=args.sampling_algo, algo_kwargs=algo_kwargs, seed=args.seed,
                           level_exposed=args.sampling_level_exposed, device=device)
    img_format = 'exr' if args.save_exr else 'png'
    sampler.evaluate(args.num_samples, config_file=args.config_file, result_dir=output_dir,
                     img_format=img_format)


if __name__ == '__main__':
    main()
