from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import json

import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch import nn

from utils.model import AE
from utils.dataset import ImageDataset
from utils.time_as_string import get_current_time_as_string
from utils.plot_utils import plot_loss


def train_ae_one_epoch(model, device, dataloader, optimizer, loss_func):
    model.train()
    epoch_loss = 0

    for batch_i, (images, background_colors) in enumerate(dataloader):
        images = images.to(device)
        optimizer.zero_grad()
        recon = model(images)
        loss = loss_func(recon, images)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= batch_i + 1

    return epoch_loss


def train_ae(model, device, train_dataloader, epochs, optimizer, loss_func, output_dir, reduce_lr=None):
    train_losses = []

    counter_obj = tqdm(list(range(epochs)))
    for epoch in counter_obj:
        train_losses.append(train_ae_one_epoch(model, device, train_dataloader, optimizer, loss_func))
        counter_obj.set_postfix_str(f'loss: {train_losses[-1]:0.3f}')

        if epoch > 0 and train_losses[-1] < min(train_losses[:-1]):
            torch.save(model.state_dict(), output_dir / 'ae_model.pth')

        if reduce_lr is not None:
            reduce_lr.step(train_losses[-1])

    loss_plot = plot_loss(train_losses)
    loss_plot.write_html(output_dir / 'loss_plot.html')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image_scale', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--loss_reduction', default='mean')
    parser.add_argument('--output_dir', default='outputs/models')
    parser.add_argument('--data_location', default='data/black_white_dataset')
    parser.add_argument('--data_background_loc', type=int, default=4)
    parser.add_argument('--data_set_loc', type=int, default=2)

    parser.add_argument('--reduce_lr', action='store_true')
    parser.add_argument('--reduce_lr_patience', type=int, default=3)
    parser.add_argument('--reduce_lr_factor', type=float, default=0.1)

    parser.add_argument('--ae_depth', type=int, default=6)
    parser.add_argument('--ae_expand_factor', type=int, default=2)
    parser.add_argument('--ae_latent_dim', type=int, default=16)
    parser.add_argument('--ae_activation_func', type=str, default='GELU')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / get_current_time_as_string()
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f'Saving output to {output_dir}')

    with open(output_dir / 'run_config.json', 'w') as f:
        json.dump(args.__dict__, f, indent=True)

    device = f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu'
    print(f'Using device {device}')

    train_dataset = ImageDataset(
        args.data_location,
        path_background_loc=args.data_background_loc,
        dataset_name='train',
        scale_images_to=args.image_scale,
        normalize=True,
        set_loc=args.data_set_loc
    )
    train_dataset.describe()

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        batch_size=args.bs
    )

    activation_functions = dict(
        ReLU=nn.ReLU,
        PReLU=nn.PReLU,
        GELU=nn.GELU,
        GLU=nn.GLU
    )

    model = AE(dict(
        depth=args.ae_depth,
        expand_factor=args.ae_expand_factor,
        latent_dim=args.ae_latent_dim,
        activation_func=activation_functions[args.ae_activation_func],
        input_dim=args.image_scale
    )).to(device)

    model.describe()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    reduce_lr = None
    if args.reduce_lr:
        reduce_lr = ReduceLROnPlateau(
            optimizer=optimizer, patience=args.reduce_lr_patience, factor=args.reduce_lr_factor, verbose=True
        )
    loss_func = nn.MSELoss(reduction=args.loss_reduction)
    train_ae(model, device, train_dataloader, args.epochs, optimizer, loss_func, output_dir, reduce_lr=reduce_lr)
