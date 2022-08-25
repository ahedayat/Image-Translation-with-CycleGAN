"""
Utilities of This Project
"""

import argparse


def get_args():
    """
    Argunments of:
        - `train.py`
        - `test.py`
    """
    parser = argparse.ArgumentParser(
        description='Arguemnt Parser of `Train` and `Evaluation` of deep neural network.')

    # Hardware
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu',
                        default=True, help='use cuda')
    parser.add_argument('-w', '--num-workers', dest='num_workers', default=1,
                        type=int, help='Number of workers for loading data')

    # CNN Backbone
    parser.add_argument('--num-residuals', dest='num_residuals', default=9, type=int,
                        help='Number of residual blocks')

    # Input Size
    parser.add_argument('--image-size', dest='image_size', default=100, type=int,
                        help='image size')

    # Data Path
    parser.add_argument('--data-df', dest='data_df', default="./datasets/Horse2Zebra/preprocessed.csv",
                        type=str, help='dataset Dataframe path')
    # - Train
    parser.add_argument('--train-base-dir', dest='train_base_dir', default="./datasets/Horse2Zebra",
                        type=str, help='train dataset base directory')
    # - validation
    parser.add_argument('--val-base-dir', dest='val_base_dir', default="./datasets/Horse2Zebra",
                        type=str, help='validation dataset base directory')
    # - Test
    parser.add_argument('--test-base-dir', dest='test_base_dir', default="./datasets/Horse2Zebra",
                        type=str, help='Test dataset base directory')

    # Loss Function
    parser.add_argument('--lambda-cycle', dest='lambda_cycle', default=0.1,
                        type=float, help="Coefficient of Cycle Loss")
    parser.add_argument('--lambda-identity', dest='lambda_identity', default=0.1,
                        type=float, help="Coefficient of Identity Loss")

    # Optimizer and its hyper-parameters
    parser.add_argument('-r', '--learning-rate', dest='lr', default=1e-3,
                        type=float, help='learning rate')
    parser.add_argument('--b1', dest='b1', default=1e-3,
                        type=float, help='beta 1')
    parser.add_argument('--b2', dest='b2', default=1e-3,
                        type=float, help='beta 2')

    # Learning Rate Scheduler
    parser.add_argument('--lr-start-decay', dest='lr_start_decay', default=100,
                        type=int, help="Number of epoch that learning rate starts decaying after that.")

    # Training Hyper-Parameter
    parser.add_argument('-e', '--epoch', dest='epochs', default=200, type=int,
                        help='number of epochs')
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=16,
                        type=int, help='batch size')

    # Saving Paths
    # Generate Images generated_images
    parser.add_argument('--generated-images', dest='generated_images', default="./generated_images",
                        type=str, help='Generated Images Path')
    parser.add_argument('--sample-interval', dest='sample_interval', default=10,
                        type=int, help='Sampling Frequency')

    # Reports Path
    parser.add_argument("--report-path",  dest='report_path', default="./report_path",
                        type=str, help='Reports Path')

    # - Check Points
    parser.add_argument('--ckpt-path', dest='ckpt_path', default="./checkpoints",
                        type=str, help='Saving check points path')
    parser.add_argument('--ckpt-prefix', dest='ckpt_prefix', default="ckpt_",
                        type=str, help='Check points is saved with this prefix')
    parser.add_argument('--ckpt-load', dest='ckpt_load', default=None,
                        type=str, help='Load check point')
    parser.add_argument('--model-saving-freq', dest='model_saving_freq', default=50,
                        type=int, help='Frequency of Saving of Model')

    # - Model Path for Loading
    parser.add_argument('--gen-x2y', dest='gen_x2y', default="./checkpoints/ckpt_gen_x2y_final",
                        type=str, help='Loading path to generator network of X->Y')
    parser.add_argument('--gen-y2x', dest='gen_y2x', default="./checkpoints/ckpt_gen_y2x_final",
                        type=str, help='Loading path to generator network of X->Y')
    parser.add_argument('--disc-x', dest='disc_x', default="./checkpoints/ckpt__disc_x_final",
                        type=str, help='Loading path to discriminator network of X')
    parser.add_argument('--disc-y', dest='disc_y', default="./checkpoints/ckpt__disc_y_final",
                        type=str, help='Loading path to discriminator network of Y')

    options = parser.parse_args()

    return options
