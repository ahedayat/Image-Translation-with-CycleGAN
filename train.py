import itertools

import torch
import torch.nn as nn

import nets
import utils
import optim
import dataloaders as data
import deeplearning as dl
from torchvision import transforms


def _main(args):
    # Hardware
    cuda = True if args.gpu and torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Input Image Parameters
    input_shape = (3, args.image_size, args.image_size)

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Data Loader
    df_path = args.data_df

    # - Train
    train_base_dir = args.train_base_dir
    train_dataloader = data.Horse2Zebra(
        df_path=df_path,
        base_dir=train_base_dir,
        transform=train_transform,
        mode="train"
    )

    # - Validation
    val_base_dir = args.val_base_dir
    val_dataloader = data.Horse2Zebra(
        df_path=df_path,
        base_dir=val_base_dir,
        transform=eval_transform,
        mode="val"
    )

    # - Test
    test_base_dir = args.test_base_dir
    test_dataloader = data.Horse2Zebra(
        df_path=df_path,
        base_dir=test_base_dir,
        transform=eval_transform,
        mode="test"
    )

    # Architecture Parameters
    num_residuals = args.num_residuals

    # Architecture
    gen_x2y = nets.Generator(input_shape[0], num_residuals)
    gen_y2x = nets.Generator(input_shape[0], num_residuals)
    disc_x = nets.Discriminator(input_shape)
    disc_y = nets.Discriminator(input_shape)

    buffer_x = nets.ReplayBuffer()
    buffer_y = nets.ReplayBuffer()

    # Loss Function
    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Migrate to GPU
    if cuda:
        gen_x2y = gen_x2y.cuda()
        gen_y2x = gen_y2x.cuda()
        disc_x = disc_x.cuda()
        disc_y = disc_y.cuda()
        criterion_gan = criterion_gan.cuda()
        criterion_cycle = criterion_cycle.cuda()
        criterion_identity = criterion_identity.cuda()

    # Weight Initialization
    gen_x2y.init_layers()
    gen_y2x.init_layers()
    disc_x.init_layers()
    disc_y.init_layers()

    # Optimizer
    optimizer_gen = torch.optim.Adam(
        itertools.chain(gen_x2y.parameters(), gen_y2x.parameters()),
        lr=args.lr,
        betas=(args.b1, args.b2)
    )
    optimizer_disc_x = torch.optim.Adam(
        disc_x.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2)
    )
    optimizer_disc_y = torch.optim.Adam(
        disc_y.parameters(),
        lr=args.lr,
        betas=(args.b1, args.b2)
    )

    # Learning Rate Scheduler
    lr_scheduler_gen = optim.lr_scheduler(
        optimizer_gen, num_epochs=args.epochs,
        decay_start_epoch=args.lr_start_decay,
        offset=0)

    lr_scheduler_disc_x = optim.lr_scheduler(
        optimizer_disc_x, num_epochs=args.epochs,
        decay_start_epoch=args.lr_start_decay,
        offset=0)

    lr_scheduler_disc_y = optim.lr_scheduler(
        optimizer_disc_y, num_epochs=args.epochs,
        decay_start_epoch=args.lr_start_decay,
        offset=0)

    # Checkpoints Address

    # Generated Images Saving Address
    generated_images_path = args.generated_images

    # Training
    gen_x2y, gen_y2x, disc_x, disc_y = dl.train(
        gen_x2y=gen_x2y,
        gen_y2x=gen_y2x,
        disc_x=disc_x,
        disc_y=disc_y,
        buffer_x=buffer_x,
        buffer_y=buffer_y,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion_gan=criterion_gan,
        criterion_cycle=criterion_cycle,
        criterion_identity=criterion_identity,
        lr_scheduler_gen=lr_scheduler_gen,
        lr_scheduler_disc_x=lr_scheduler_disc_x,
        lr_scheduler_disc_y=lr_scheduler_disc_y,
        optim_gen=optimizer_gen,
        optim_disc_x=optimizer_disc_x,
        optim_disc_y=optimizer_disc_y,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        num_epoch=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_interval=args.sample_interval,
        val_saving_root_path=generated_images_path,
        cuda=cuda,
        Tensor=Tensor,
        ckpt_path=args.ckpt_path,
        ckpt_prefix=args.ckpt_prefix,
        model_saving_freq=args.model_saving_freq,
        reports_path=args.report_path
    )

    # Saving Results
    ckpt_path, ckpt_prefix = args.ckpt_path, args.ckpt_prefix

    # Saving 'gen_x2y'
    nets.save(
        file_path=ckpt_path,
        file_name=f"{ckpt_prefix}_gen_x2y_final.ckpt",
        model=gen_x2y,
        optimizer=optimizer_gen
    )
    # Saving 'gen_y2x'
    nets.save(
        file_path=ckpt_path,
        file_name=f"{ckpt_prefix}_gen_y2x_final.ckpt",
        model=gen_y2x,
        optimizer=optimizer_gen
    )
    # Saving 'disc_x'
    nets.save(
        file_path=ckpt_path,
        file_name=f"{ckpt_prefix}_disc_x_final.ckpt",
        model=disc_x,
        optimizer=optimizer_disc_x
    )
    # Saving 'disc_x'
    nets.save(
        file_path=ckpt_path,
        file_name=f"{ckpt_prefix}_disc_y_final.ckpt",
        model=disc_y,
        optimizer=optimizer_disc_y
    )


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
