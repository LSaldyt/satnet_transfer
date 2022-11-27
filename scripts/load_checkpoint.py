import numpy as np
import satnet
from time import time
import torch
from satnet_transfer.settings import Settings
from satnet_transfer.dataset.base_dataset import SATDataset
from satnet_transfer.loop import loop
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt

from .tri_modal_gauss import tri_modal_gauss

dimension_lookup = {
    'bw_larg.c' : 3016,
    'aim-200-2_0-yes1-4' : 200,
    'huge' : 459,
    'hanoi5' : 1931,
    'CBS_k3_n100_m403_b10_930' : 100,
    'aim-50-2_0-yes1-4' : 50,
    'aim-100-2_0-yes1-2' : 100,
    'bw_large.b' : 1087,
    'medium' : 116,
    'CBS_k3_n100_m403_b10_999_solutions' : 100,
    'aim-100-3_4-yes1-1' : 100,
    'CBS_k3_n100_m403_b10_919' : 100,
    'logistics.a' : 828,
    'logistics.d' : 4713,
    'enddr2-10-by-5-8' : 21000,
    'aim-100-6_0-yes1-1' : 100,
    'bw_large.d' : 6325,
    'logistics.c' : 1141,
    'hanoi4' : 718,
    'CBS_k3_n100_m403_b10_420' : 100,
    'bw_large.a' : 459,
    'logistics.b' : 843,
    'anomaly' : 48
        }

# See https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
def run():
    prob = 'logistics.d'
    e = 9
    checkpoint_dir  = Path(f'interpretability_experiment')
    checkpoint_path = checkpoint_dir / Path(f'{prob}_mask.pt_{e}')

    # Initialize a basic SATNet with the correct dimensions for the problem
    n   = dimension_lookup[prob]
    m   = 1000
    aux = 1000
    sat = satnet.SATNet(n, m, aux, tri_modal=True)

    # Load the saved checkpoint based on prob and epoch vars
    checkpoint = torch.load(checkpoint_path)
    sat.load_state_dict(checkpoint['model_state_dict'])
    # Can also load optimizer and epoch

    # Plot heatmaps
    sns.set_style('whitegrid')
    hmap_kwargs = dict(xlabel='Input Variables', ylabel='SDP Clauses')

    # Normal S matrix heatmap
    S = sat.S.detach().numpy()
    ax = sns.heatmap(S)
    ax.set(title='Normal S Matrix', **hmap_kwargs)
    plt.show()
    plt.savefig(f'plots/normal_s_matrix_{prob}_{e}.png')

    # Tri Modal S matrix heatmap
    # S_tri = tri_modal_gauss(S, 0.1) / 4 # Hacked in normalization :)
    S_tri = tri_modal_gauss(S, 0.2) / 4 # Hacked in normalization :)
    ax = sns.heatmap(S_tri)
    ax.set(title='Tri Modal Gaussian Activation', **hmap_kwargs)
    plt.show()
    plt.savefig(f'plots/tri_modal_gaussian_s_matrix_{prob}_{e}.png')

    # Distribution plots
    dist_kwargs = dict(xlabel='Range', ylabel='Probability')
    ax = sns.displot(S.ravel())
    ax.set(title='Normal S Matrix Distribution', **dist_kwargs)
    plt.show()
    plt.savefig(f'plots/normal_s_dist_{prob}_{e}.png')

    ax = sns.displot(S_tri.ravel())
    ax.set(title='Tri Modal S Matrix Distribution', **dist_kwargs)
    plt.show()
    plt.savefig(f'plots/tri_modal_s_dist_{prob}_{e}.png')
