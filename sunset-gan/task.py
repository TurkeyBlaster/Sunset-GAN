from .model import DiscoGAN

from tensorflow.compat.v1.logging import set_verbosity
from argparse import ArgumentParser

def get_args():

    init_parser = ArgumentParser(description='Parses arguments for model construction (and sub-model initialization)')
    train_parser = ArgumentParser(description='Parses arguments for model training (and interim visualization), such as hyperparameters')
    meta_parser = ArgumentParser(description='Parses arguments for model saving and other script processes')

    init_parser.add_argument(
        '--input-shape',
        required = True,
        nargs = 3,
        type = int,
        help = 'Images dimensions\nRequired'
    )

    init_parser.add_argument(
        '--project-name',
        type = str,
        help = 'Name of gcloud project\nNot required'
    )

    init_parser.add_argument(
        '--key',
        type = str,
        help = 'Filepath to gcloud project keys\nNot required'
    )

    init_parser.add_argument(
        '--d-fp',
        type = str,
        help = 'Path to existing discriminator\nNot required'
    )

    init_parser.add_argument(
        '--g-fp',
        type = str,
        help = 'Path to existing discriminator\nNot required'
    )

    init_parser.add_argument(
        '--d-opt',
        type = str,
        default = 'Adam',
        help = 'Discriminator optimizer\nDefault: Adam'
    )

    init_parser.add_argument(
        '--g-opt',
        type = str,
        default = 'Adam',
        help = 'Generator optimizer\nDefault: \'Adam\''
    )

    init_parser.add_argument(
        '--d-lr',
        type = float,
        default = 4e-4,
        help = 'Discriminator learning rate\nDefault: 0.0004'
    )

    init_parser.add_argument(
        '--g-lr',
        type = float,
        default = 4e-4,
        help = 'Generator learning rate\nDefault: 0.0004'
    )

    init_parser.add_argument(
        '--d-opt-params',
        nargs = '*',
        help = 'Parameters for discriminator optimizer\nDefault: {}'
    )

    init_parser.add_argument(
        '--g-opt-params',
        nargs = '*',
        help = 'Parameters for generator optimizer\nDefault: {}'
    )

    init_parser.add_argument(
        '--print-summaries',
        type = bool,
        help = 'Whether to print the model summaries'
    )

    train_parser.add_argument(
        '--data-dir',
        required = True,
        type = str,
        help = 'Directory for the training data\nRequired'
    )

    train_parser.add_argument(
        '--epochs',
        type = int,
        default = 1,
        help = 'Number of training epochs\nDefault: 1'
    )

    train_parser.add_argument(
        '--batch_size',
        type = int,
        default = 16,
        help = 'Batch size of data for model training\nDefault: 128'
    )

    train_parser.add_argument(
        '--d-initial-reg',
        type = float,
        default = 1e-2,
        help = 'Initial l2 regularization of the discriminator\n \
                (Will decay as timesteps increase)\n \
                Default: 0.01'
    )

    train_parser.add_argument(
        '--g-initial-reg',
        type = float,
        default = 1e-2,
        help = 'Initial l2 regularization of the generator\n \
                (Will decay as timesteps increase)\n \
                Default: 0.01'
    )

    train_parser.add_argument(
        '--d-min-reg',
        type = float,
        default = 1e-4,
        help = 'Minimum l2 regularization of the discriminator\nDefault: 0.0001'
    )

    train_parser.add_argument(
        '--g-min-reg',
        type = float,
        default = 1e-4,
        help = 'Minimum l2 regularization of the generator\nDefault: 0.0001'
    )

    train_parser.add_argument(
        '--max-q-size',
        type = int,
        default = 25,
        help = 'Number of past states to be saved (if reached)\n \
                Equal to the maximum length of the queue\n \
                Default: 25'
    )

    train_parser.add_argument(
        '--q-update-inc',
        type = int,
        default = 10,
        help = 'Number of timesteps in between queue updates\nDefault: 10'
    )

    train_parser.add_argument(
        '--num-plots',
        type = int,
        default = 5,
        help = 'Number of images to plot (generator demonstration)\nDefault: 5'
    )

    train_parser.add_argument(
        '--plot-dir',
        type = str,
        default = '',
        help = 'Directory in which to save plots\nDefault: \'\''
    )

    train_parser.add_argument(
        '--plot-tstep',
        type = int,
        default = 1,
        help = 'Number of timesteps in between generator plotting\nDefault: 1'
    )

    meta_parser.add_argument(
        '--save-dir',
        required = True,
        type = str,
        help = 'Directory in which to save models\nRequired'
    )

    init_args, _ = init_parser.parse_known_args()
    train_args, _ = train_parser.parse_known_args()
    meta_args, _ = meta_parser.parse_known_args()

    return init_args, train_args, meta_args

if __name__ == '__main__':
    
    set_verbosity('INFO')

    init_args, train_args, meta_args = get_args()
    
    gan = DiscoGAN(**vars(init_args))
    gan.train(**vars(train_args))
    gan.save_models(meta_args.save_dir)
