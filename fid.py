import os
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

    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Data Loader
    df_path = args.data_df

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

    # # Optimizer
    # optimizer_gen = torch.optim.Adam(
    #     itertools.chain(gen_x2y.parameters(), gen_y2x.parameters()),
    #     lr=args.lr,
    #     betas=(args.b1, args.b2)
    # )

    # Migrate to GPU
    if cuda:
        gen_x2y = gen_x2y.cuda()
        gen_y2x = gen_y2x.cuda()

    # Load Model
    # gen_x2y, optimizer_gen = nets.load(args.gen_x2y, gen_x2y, optimizer_gen)
    # gen_y2x, optimizer_gen = nets.load(args.gen_y2x, gen_y2x, optimizer_gen)
    gen_x2y, _ = nets.load(args.gen_x2y, gen_x2y, None)
    gen_y2x, _ = nets.load(args.gen_y2x, gen_y2x, None)

    # Paths
    real_x_path = os.path.join(test_base_dir, "testA")
    fake_x_path = os.path.join(test_base_dir, "fakeA")
    real_y_path = os.path.join(test_base_dir, "testB")
    fake_y_path = os.path.join(test_base_dir, "fakeB")

    # Evaluation
    fid_x, fid_y = dl.fid_score(
        gen_x2y=gen_x2y,
        gen_y2x=gen_y2x,
        eval_dataloader=test_dataloader,
        real_x_path=real_x_path,
        fake_x_path=fake_x_path,
        real_y_path=real_y_path,
        fake_y_path=fake_y_path,
        Tensor=Tensor,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )

    print(f"FID -> X: {fid_x}, Y: {fid_y}")


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
