from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.nn.functional import interpolate
import torch as th

from diffmat import MaterialGraphTranslator as MGT, config_logger
from diffmat.core.io import read_image
from diffmat.core.util import FILTER_OFF
from diffmat.optim import ParamPredictor


def main():
    """Train or evaluate a parameter prediction network of a differentiable procedural material
    graph.
    """
    # Default paths
    default_result_dir = Path(__file__).resolve().parents[0] / 'result'

    # Command line argument parser
    prog_description = 'Parameter prediction network training and evaluation'
    parser = argparse.ArgumentParser(description=prog_description)

    ## Operation mode ('train' or 'eval')
    parser.add_argument('mode', choices=['train', 'eval'],
                        help="Operation mode ('train' or 'eval')")

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

    ## Network architecture
    parser.add_argument('-d', '--num-layers', metavar='NUM', type=int, default=3,
                        help='Number of fully-connected layers in MLP')
    parser.add_argument('-wp', '--layer-size-param-mult', metavar='RATIO', type=int, default=3,
                        help='Width of FC layers as a multiplier of graph parameter number')
    parser.add_argument('-wm', '--layer-size-max', metavar='SIZE', type=int, default=960,
                        help='Maximum width of FC layers')
    parser.add_argument('-tl', '--td-pyramid-level', metavar='LEVEL', type=int, default=2,
                        help='Image pyramid depth when calculating the texture descriptor')

    ## Network training and evaluation
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random sampler seed')
    parser.add_argument('-n', '--num-epochs', type=int, default=150,
                        help='Number of epochs in total')
    parser.add_argument('-nt', '--num-epochs-td', type=int, default=100,
                        help='Number of epochs after which texture descriptor loss is introduced')
    parser.add_argument('-m', '--epoch-iters', type=int, default=100,
                        help='Number of iterations/batches in each epoch')
    parser.add_argument('-b', '--batch-size', type=int, default=5,
                        help='Batch size before introducing texture descriptor loss')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('-lc', '--lr-decay-coeff', type=float, default=0.5,
                        help='Learning rate decay coefficient')
    parser.add_argument('-tw', '--td-loss-weight', type=float, default=0.1,
                        help='Weight of the texture descriptor loss term')
    parser.add_argument('-si', '--save-interval', type=int, default=20,
                        help='Number of epochs between two checkpoints')
    parser.add_argument('-le', '--load-checkpoint-epoch', type=int, default=-1,
                        help='Epoch number to load checkpoint from')
    parser.add_argument('-ld', '--load-checkpoint-dir', default='',
                        help="The folder where checkpoints are stored")
    parser.add_argument('-lve', '--filter-exposed', type=int, default=FILTER_OFF,
                        help='Exposed parameter training level')
    parser.add_argument('-lvd', '--filter-generator', type=int, default=FILTER_OFF,
                        help='Discrete parameter training level')
    parser.add_argument('-v', '--save-validation-imgs', action='store_true',
                        help='Save input and predicted images during validation')

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

    # Get the translated graph object
    device = 'cpu' if args.cpu else 'cuda'
    graph = translator.translate(
        seed=args.graph_seed, normal_format=args.normal_format, external_input_folder=ext_input_dir,
        device=device)
    graph.compile()

    # Initialize the parameter predictor
    arch_kwargs = {v: getattr(args, v) for v in
                   ('num_layers', 'layer_size_param_mult', 'layer_size_max', 'td_pyramid_level')}
    sampler_kwargs = {
        **{v: getattr(args, v) for v in ('seed', 'filter_exposed', 'filter_generator')},
        'algo_kwargs': {'min': -0.15, 'max': 0.15, 'mu': 0.0, 'sigma': 0.03}
    }

    predictor = ParamPredictor(
        graph, arch_kwargs=arch_kwargs, sampler_kwargs=sampler_kwargs, device=device)

    # Training mode
    if args.mode == 'train':

        # Define training plans
        num_epochs = min(args.num_epochs, args.num_epochs_td)
        num_epochs = num_epochs, max(args.num_epochs - args.num_epochs_td, 0)
        num_epochs = num_epochs[0] if num_epochs[1] == 0 else num_epochs
        td_loss_weight = 0.0, args.td_loss_weight

        # Start training
        start_epoch = max(args.load_checkpoint_epoch, 0)
        load_checkpoint = args.load_checkpoint_epoch >= 0
        img_format = 'exr' if args.exr else 'png'
        output_dir = result_dir / (args.output_dir_name or 'network_train')

        predictor.train(num_epochs=num_epochs, epoch_iters=args.epoch_iters,
                        td_loss_weight=td_loss_weight, lr=args.lr,
                        lr_decay_coeff=args.lr_decay_coeff, batch_size=args.batch_size,
                        start_epoch=start_epoch, load_checkpoint=load_checkpoint,
                        load_checkpoint_dir=args.load_checkpoint_dir,
                        save_interval=args.save_interval, result_dir=output_dir,
                        save_validation_imgs=args.save_validation_imgs, img_format=img_format)

    # Evaluation mode
    else:

        # Path to an input image must be provided
        if not args.input_image:
            raise ValueError('Path to an input image must be provided in evaluation mode')

        # Load trained parameters from a checkpoint file
        if args.load_checkpoint_epoch >= 0:
            load_dir = Path(args.load_checkpoint_dir) if args.load_checkpoint_dir else \
                       result_dir / 'network_train' / 'checkpoints'
            state_dict = th.load(load_dir / f'epoch_{args.load_checkpoint_epoch}.pth')
            predictor.model.load_state_dict(state_dict['model'])

        # Read the specified input image from local file (e.g., real-world target) and resize it
        # to the target optimization size
        img_size = (1 << args.res, 1 << args.res)
        input_img = read_image(args.input_image)[:3].unsqueeze(0)
        input_img = interpolate(
            input_img, size=img_size, mode='bilinear', align_corners=False)

        # Predict parameters using the network and save the result (SVBRDF maps, rendered image,
        # parameters)
        output_dir = result_dir / (args.output_dir_name or 'network_pred')
        predictor.evaluate(input_img, save_result=True, result_dir=output_dir)


if __name__ == '__main__':
    main()
