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
    # files_A_index = [0, 12, 15, 27, 33, 41, 45, 51,
    #                  54, 62, 63, 64, 66, 79, 84, 85, 91, 93, 94]
    files_A_index = [94, 12, 62, 66, 85, 51]
    # files_B_index = [1, 3, 10, 11, 15, 19, 23, 32, 34, 37,
    #                  38, 52, 53, 55, 75, 99, 102, 103, 105, 127, 129, 134]
    files_B_index = [102, 105, 127, 134, 34, 38, 53, 34]

    files_A = list()
    for ix in files_A_index:
        files_A.append(test_dataloader.files_A[ix])

    files_B = list()
    for ix in files_B_index:
        files_B.append(test_dataloader.files_B[ix])

    test_dataloader.files_A = files_A
    test_dataloader.files_B = files_B

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

    # Generated Images Saving Address
    generated_images_path = args.generated_images

    # Evaluation
    dl.eval(
        gen_x2y=gen_x2y,
        gen_y2x=gen_y2x,
        eval_dataloader=test_dataloader,
        Tensor=Tensor,
        saving_root_path=generated_images_path,
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
