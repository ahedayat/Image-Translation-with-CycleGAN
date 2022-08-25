import os

import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import transforms

import nets
import pytorch_fid


def train(
    gen_x2y,
    gen_y2x,
    disc_x,
    disc_y,
    buffer_x,
    buffer_y,
    train_dataloader,
    val_dataloader,
    criterion_gan,
    criterion_cycle,
    criterion_identity,
    lr_scheduler_gen,
    lr_scheduler_disc_x,
    lr_scheduler_disc_y,
    optim_gen,
    optim_disc_x,
    optim_disc_y,
    lambda_cycle=10,
    lambda_identity=5,
    num_epoch=200,
    batch_size=16,
    num_workers=2,
    Tensor=torch.Tensor,
    cuda=False,
    sample_interval=10,
    val_saving_root_path="./generated_images",
    ckpt_path="./checkpoints",
    ckpt_prefix="ckpt_",
    model_saving_freq=50,
    reports_path="./reports/"
):
    """
        Training Function
    """
    train_dataloader = DataLoader(
        train_dataloader,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_dataloader = DataLoader(
        val_dataloader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    for e in range(num_epoch):
        df_report = pd.DataFrame(
            columns=[
                "epoch",
                "batch_id",
                "batch_num",
                "learning_rate",
                "loss_id_x",
                "loss_id_y",
                "loss_identity",
                "loss_gan_y2x",
                "loss_gan_x2y",
                "loss_gan",
                "loss_cycle_x",
                "loss_cycle_y",
                "loss_cycle",
                "loss_generation",
                "loss_disc_real_x",
                "loss_disc_fake_x",
                "loss_disc_real_y",
                "loss_disc_fake_y",
                "loss_disc"
            ]
        )
        with tqdm(train_dataloader) as t_loader:
            for batch_id, (X, Y) in enumerate(t_loader):
                t_loader.set_description(f"Training @ Epoch {e}")

                X = Variable(X.type(Tensor))
                Y = Variable(Y.type(Tensor))

                valid = Variable(
                    Tensor(np.ones((X.size(0), *disc_x.out_size))).type(Tensor),
                    requires_grad=False
                )
                fake = Variable(
                    Tensor(np.zeros((X.size(0), *disc_x.out_size))).type(Tensor),
                    requires_grad=False
                )

                #------------------#
                # Train Geneartors #
                #------------------#
                gen_x2y.train()
                gen_y2x.train()

                optim_gen.zero_grad()

                # identity loss
                loss_id_x = criterion_identity(gen_y2x(X), X)
                loss_id_y = criterion_identity(gen_x2y(Y), Y)
                loss_identity = (loss_id_x + loss_id_y) / 2

                # Generating fake X and Y
                fake_X = gen_y2x(Y)
                fake_Y = gen_x2y(X)

                # GAN Loss
                loss_gan_y2x = criterion_gan(disc_x(fake_X), valid)
                loss_gan_x2y = criterion_gan(disc_y(fake_Y), valid)
                loss_gan = (loss_gan_x2y + loss_gan_y2x) / 2

                # Reconstruction of X and Y
                reconstructed_X = gen_y2x(fake_Y)
                reconstructed_Y = gen_x2y(fake_X)

                # Cycle Loss
                loss_cycle_X = criterion_cycle(reconstructed_X, X)
                loss_cycle_Y = criterion_cycle(reconstructed_Y, Y)
                loss_cycle = (loss_cycle_X + loss_cycle_Y)/2

                # Total Loss
                loss_gen = loss_gan + lambda_cycle * loss_cycle + lambda_identity * loss_identity

                # Backpropagation and Updating Weights for Generator
                loss_gen.backward()
                optim_gen.step()

                #--------------------------#
                # Train Discriminator of X #
                #--------------------------#
                optim_disc_x.zero_grad()

                # Real Loss for X
                loss_real_X = criterion_gan(disc_x(X), valid)

                # Fake Loss for X
                _fake_X = buffer_x.push_pop(fake_X).detach()
                loss_fake_X = criterion_gan(disc_x(_fake_X), fake)

                # Total Loss of Discriminator of X
                loss_disc_X = (loss_real_X + loss_fake_X) / 2

                # Backpropagation and Updating Weights for Discriminator of X
                loss_disc_X.backward()
                optim_disc_x.step()

                #--------------------------#
                # Train Discriminator of Y #
                #--------------------------#
                optim_disc_y.zero_grad()

                # Real Loss for Y
                loss_real_Y = criterion_gan(disc_y(Y), valid)

                # Fake Loss for Y
                _fake_Y = buffer_y.push_pop(fake_Y).detach()
                loss_fake_Y = criterion_gan(disc_y(_fake_Y), fake)

                # Total Loss of Discriminator of X
                loss_disc_Y = (loss_real_Y + loss_fake_Y) / 2

                # Backpropagation and Updating Weights for Discriminator of X
                loss_disc_Y.backward()
                optim_disc_y.step()

                loss_disc = (loss_disc_X + loss_disc_Y) / 2

                #--------------#
                # Progress Bar #
                #--------------#
                t_loader.set_postfix(
                    loss_disc="{:.3f}".format(loss_disc.item()),
                    loss_gen="{:.3f}".format(loss_gen.item()),
                    loss_gan="{:.3f}".format(loss_gan.item()),
                    loss_cycle="{:.3f}".format(loss_cycle.item()),
                    loss_identity="{:.3f}".format(loss_identity.item()),
                    lr="{:.5e}".format(optim_disc_x.param_groups[0]['lr'])
                )

                batches_done = e * len(train_dataloader) + batch_id

                if batches_done % sample_interval == 0:
                    eval_generator_train(
                        gen_x2y=gen_x2y,
                        gen_y2x=gen_y2x,
                        eval_dataloader=val_dataloader,
                        saving_root_path=val_saving_root_path,
                        saving_name=f"validation_epoch_{e}",
                        Tensor=Tensor
                    )

                df_report = df_report.append(
                    {
                        "epoch": e,
                        "batch_id": batch_id,
                        "batch_num": X.shape[0],
                        "learning_rate": optim_disc_x.param_groups[0]['lr'],
                        "loss_id_x": loss_id_x.item(),
                        "loss_id_y": loss_id_y.item(),
                        "loss_identity": loss_identity.item(),
                        "loss_gan_x2y": loss_gan_x2y.item(),
                        "loss_gan_y2x": loss_gan_y2x.item(),
                        "loss_gan": loss_gan.item(),
                        "loss_cycle_x": loss_cycle_X.item(),
                        "loss_cycle_y": loss_cycle_Y.item(),
                        "loss_cycle": loss_cycle.item(),
                        "loss_generation": loss_gen.item(),
                        "loss_disc_real_x": loss_real_X.item(),
                        "loss_disc_fake_x": loss_fake_X.item(),
                        "loss_disc_real_y": loss_real_Y.item(),
                        "loss_disc_fake_y": loss_fake_Y.item(),
                        "loss_disc": loss_disc.item()
                    }, ignore_index=True)

        # Saving Report
        df_report.to_csv(os.path.join(reports_path, f"train_{e}.csv"))

        # Learning Rate Scheduling
        lr_scheduler_gen.step()
        lr_scheduler_disc_x.step()
        lr_scheduler_disc_y.step()

        if e % model_saving_freq == 0:
            # Saving 'gen_x2y'
            nets.save(
                file_path=ckpt_path,
                file_name=f"{ckpt_prefix}_gen_x2y_{e}.ckpt",
                model=gen_x2y,
                optimizer=optim_gen
            )
            # Saving 'gen_y2x'
            nets.save(
                file_path=ckpt_path,
                file_name=f"{ckpt_prefix}_gen_y2x_{e}.ckpt",
                model=gen_y2x,
                optimizer=optim_gen
            )
            # Saving 'disc_x'
            nets.save(
                file_path=ckpt_path,
                file_name=f"{ckpt_prefix}_disc_x_{e}.ckpt",
                model=disc_x,
                optimizer=optim_disc_x
            )
            # Saving 'disc_x'
            nets.save(
                file_path=ckpt_path,
                file_name=f"{ckpt_prefix}_disc_y_{e}.ckpt",
                model=disc_y,
                optimizer=optim_disc_y
            )

    return gen_x2y, gen_y2x, disc_x, disc_y


def eval_generator_train(
    gen_x2y,
    gen_y2x,
    eval_dataloader,
    Tensor=torch.Tensor,
    saving_root_path="./generated_images",
    saving_name="generated"
):
    """
        Evaluation of Generator Networks for One Batch
    """
    X, Y = next(iter(eval_dataloader))

    X = Variable(X.type(Tensor))
    Y = Variable(Y.type(Tensor))

    batch_size = X.shape[0]

    gen_x2y.eval()
    gen_y2x.eval()

    fake_X = gen_y2x(Y)
    fake_Y = gen_x2y(X)

    X = make_grid(X, nrow=batch_size, normalize=True)
    Y = make_grid(Y, nrow=batch_size, normalize=True)
    fake_X = make_grid(fake_X, nrow=batch_size, normalize=True)
    fake_Y = make_grid(fake_Y, nrow=batch_size, normalize=True)

    image_grid = torch.cat((X, fake_Y, Y, fake_X), 1)

    path = os.path.join(saving_root_path, f"{saving_name}.png")

    save_image(image_grid, path, normalize=False)


def eval_generator(
    gen_x2y,
    gen_y2x,
    eval_dataloader,
    Tensor=torch.Tensor,
    saving_root_path="./evaluation_images",
    num_workers=2,
    batch_size=8
):
    """
        Evaluation of Generator Networks
    """

    eval_dataloader = DataLoader(
        eval_dataloader,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    with tqdm(eval_dataloader) as t_loader:
        for batch_id, (X, Y) in enumerate(t_loader):
            t_loader.set_description(f"Evaluation of Generators")

            X = Variable(X.type(Tensor))
            Y = Variable(Y.type(Tensor))

            batch_size = X.shape[0]

            gen_x2y.eval()
            gen_y2x.eval()

            fake_X = gen_y2x(Y)
            fake_Y = gen_x2y(X)

            X = make_grid(X, nrow=batch_size, normalize=True)
            Y = make_grid(Y, nrow=batch_size, normalize=True)
            fake_X = make_grid(fake_X, nrow=batch_size, normalize=True)
            fake_Y = make_grid(fake_Y, nrow=batch_size, normalize=True)

            image_grid = torch.cat((X, fake_Y, Y, fake_X), 1)

            path = os.path.join(saving_root_path, f"{batch_id}.png")

            save_image(image_grid, path, normalize=False)


def fid_score(
        gen_x2y,
        gen_y2x,
        eval_dataloader,
        real_x_path,
        fake_x_path,
        real_y_path,
        fake_y_path,
        Tensor=torch.Tensor,
        num_workers=2,
        batch_size=8,
        tqdm_description=None):
    """
        Calculating FID Score
    """

    eval_dataloader = DataLoader(
        eval_dataloader,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    to_pil = transforms.ToPILImage()

    # Saving fake xs and ys
    with tqdm(eval_dataloader) as t_loader:
        for batch_id, (x, y) in enumerate(t_loader):
            if tqdm_description is None:
                t_loader.set_description(f"Saving Fake Images")
            else:
                t_loader.set_description(
                    f"{tqdm_description} (Saving Fake Images)")

            x = Variable(x.type(Tensor))
            y = Variable(y.type(Tensor))

            fake_y = gen_x2y(x).squeeze(dim=0)
            fake_x = gen_y2x(y).squeeze(dim=0)

            fake_x = to_pil(fake_x).save(
                os.path.join(fake_x_path, f"{batch_id}.png"))

            fake_y = to_pil(fake_y).save(
                os.path.join(fake_y_path, f"{batch_id}.png"))

    fid_score_x = pytorch_fid.fid(real_x_path,
                                  fake_x_path,
                                  batch_size=batch_size,
                                  device=None)

    fid_score_y = pytorch_fid.fid(real_y_path,
                                  fake_y_path,
                                  batch_size=batch_size,
                                  device=None)

    return fid_score_x, fid_score_y
