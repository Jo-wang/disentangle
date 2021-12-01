import argparse
import os
import sys
import json
import os
import re

import numpy as np
import torch
from disvae.models.vae import VAEBase
from utils.helpers import FormatterNoDuplicate, check_bounds, set_seed
from utils.visualize import Visualizer
from utils.viz_helpers import get_samples
from main import RES_DIR
from disvae.utils.modelIO import load_model, load_metadata
from load_mnist import load_mnist
import torch
import torch.nn
import argparse
from torch.nn import DataParallel
# from __future__ import print_function, absolute_import
import os
from utils.datasets import MNIST
PLOT_TYPES = ['generate-samples', 'data-samples', 'reconstruct', "traversals",
              'reconstruct-traverse', "gif-traversals", "all"]


def load_mnist_one(idx=1):
    """
    load one image with a specific idx from mnist dataset
    """
    td, tl, _, _ = load_mnist()
    img = td[idx][0]
    label = 0
    return img, label


def load_model(model_dir, model):
    checkpoint = torch.load(model_dir)
    vae = checkpoint['model']
    model.load_state_dict(vae)
    


def main(args):
    """Main function for plotting fro pretrained models.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    device_ids = [0]
    set_seed(args.seed)
    model_dir = args.model_dir
    model = VAEBase(args)
    # model = DataParallel(model, device_ids).cuda()
    # meta_data = load_metadata(model_dir)
    load_model(model_dir, model)
    # model = DataParallel(model, device_ids)
    model.cuda()
    model.eval()  # don't sample from latent: use mean
    dataset = 'mnist'
    viz = Visualizer(model=model,
                     model_dir=model_dir,
                     dataset=dataset,
                     max_traversal=args.max_traversal,
                     loss_of_interest='kl_loss_',
                     upsample_factor=args.upsample_factor)
    size = (args.n_rows, args.n_cols)
    # same samples for all plots: sample max then take first `x`data  for all plots
    num_samples = args.n_cols * args.n_rows
    samples = get_samples(dataset, num_samples, idcs=args.idcs)   # 8,3,32,32

    if "all" in args.plots:
        args.plots = [p for p in PLOT_TYPES if p != "all"]

    for plot_type in args.plots:
        if plot_type == 'generate-samples':
            viz.generate_samples(size=size)
        elif plot_type == 'data-samples':
            viz.data_samples(samples, size=size)
        elif plot_type == "reconstruct":
            viz.reconstruct(samples, size=size)
        elif plot_type == 'traversals':
            viz.traversals(data=samples[0:1, ...] if args.is_posterior else None,
                           n_per_latent=args.n_cols,
                           n_latents=args.n_rows,
                           is_reorder_latents=False)
        elif plot_type == "reconstruct-traverse":
            viz.reconstruct_traverse(samples,
                                     is_posterior=args.is_posterior,
                                     n_latents=args.n_rows,
                                     n_per_latent=args.n_cols,
                                     is_show_text=args.is_show_loss)
        elif plot_type == "gif-traversals":
            viz.gif_traversals(samples[:args.n_cols, ...], n_latents=args.n_rows)
        else:
            raise ValueError("Unkown plot_type={}".format(plot_type))
        
def get_args():
    description = ""
    parser = argparse.ArgumentParser(description=description)
    # working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--name', type=str, default="VaeBase_Mnist",
                        help="Name of the model for storing and loading purposes.")
    parser.add_argument("--plots", type=str, default=["reconstruct-traverse"], choices=PLOT_TYPES,  #reconstruct-traverse
                        help="List of all plots to generate. `generate-samples`: random decoded samples. `data-samples` samples from the dataset. `reconstruct` first rnows//2 will be the original and rest will be the corresponding reconstructions. `traversals` traverses the most important rnows dimensions with ncols different samples from the prior or posterior. `reconstruct-traverse` first row for original, second are reconstructions, rest are traversals. `gif-traversals` grid of gifs where rows are latent dimensions, columns are examples, each gif shows posterior traversals. `all` runs every plot.")
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed. Can be `None` for stochastic behavior.')
    parser.add_argument('-r', '--n-rows', type=int, default=2,
                        help='The number of rows to visualize (if applicable).')
    parser.add_argument('-c', '--n-cols', type=int, default=4,
                        help='The number of columns to visualize (if applicable).')
    parser.add_argument('-t', '--max-traversal', default=1,
                        type=lambda v: check_bounds(v, lb=0, is_inclusive=False,
                                                    type=float, name="max-traversal"),
                        help='The maximum displacement induced by a latent traversal. Symmetrical traversals are assumed. If `m>=0.5` then uses absolute value traversal, if `m<0.5` uses a percentage of the distribution (quantile). E.g. for the prior the distribution is a standard normal so `m=0.45` corresponds to an absolute value of `1.645` because `2m=90%%` of a standard normal is between `-1.645` and `1.645`. Note in the case of the posterior, the distribution is not standard normal anymore.')
    parser.add_argument('-i', '--idcs', type=int, nargs='+', default=[],
                        help='List of indices to of images to put at the begining of the samples.')
    parser.add_argument('-u', '--upsample-factor', default=1,
                        type=lambda v: check_bounds(v, lb=1, is_inclusive=True,
                                                    type=int, name="upsample-factor"),
                        help='The scale factor with which to upsample the image (if applicable).')
    parser.add_argument('--is-show-loss', default = False, action='store_true',
                        help='Displays the loss on the figures (if applicable).')
    parser.add_argument('--is-posterior', default=True, action='store_true',
                        help='Traverses the posterior instead of the prior.')
    parser.add_argument('--num_domains', default=5)
    parser.add_argument('--domain_in_features', default=100)
    parser.add_argument('--in_features', default=2048)
    parser.add_argument('--model_dir', default="/home/zixin/MSOUDA-1/checkpoints/1D-digits_tar-mnistm_A-vae_B-20_O-Pseudo_best.pth.tar")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # args = parse_arguments([])
    # main(args)
    
    # parser.add_argument('--domain_labels', default=5)
    # parser.parse_args('VAEbase_mnist reconstruct-traverse -c 5 -r 1 -t 2 --is-posterior'.split() )
    with torch.no_grad():
        args = get_args()
        main(args)


