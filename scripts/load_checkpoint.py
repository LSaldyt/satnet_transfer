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
    # prob = 'logistics.d'
    # prob = 'CBS_k3_n100_m403_b10_930'
    e = 9

    checkpoint_dir  = Path(f'interpretability_experiment')
    plot_dir = Path('plots/tri_modal_activations')

    for prob in dimension_lookup:
        checkpoint_path = checkpoint_dir / Path(f'{prob}_mask.pt_{e}')
        if not checkpoint_path.is_file():
            print(f'Skipping {prob} since it is empty')
            continue

        # Initialize a basic SATNet with the correct dimensions for the problem
        n   = dimension_lookup[prob]
        m   = 1000
        aux = 1000
        sat = satnet.SATNet(n, m, aux, tri_modal=True)

        # Load the saved checkpoint based on prob and epoch vars
        checkpoint = torch.load(checkpoint_path)
        sat.load_state_dict(checkpoint['model_state_dict'])
        # Can also load optimizer and epoch

        S_normal = sat.S.detach().numpy()
        var = 0.1 # Ending variance after annealing
        S_tri = tri_modal_gauss(S_normal, var) / 4 # Hacked in normalization :)

        # Plot heatmaps
        sns.set_style('whitegrid')
        hmap_kwargs = dict(ylabel='Input Variables', xlabel='SDP Clauses')
        dist_kwargs = dict(xlabel='Range', ylabel='Probability')
        marg_kwargs = dict(xlabel='Input Variable', ylabel='Weight', xticklabels=[])
        bar_kwargs  = dict(saturation=1.0, width=5.0)

        def save(name, label, show=False, **kwargs):
            plt.savefig(
                plot_dir / f'{label.lower()}_{name}_{prob}_{e}.png',
                **kwargs)
            if show:
                plt.show()

        for S, label in [(S_normal, 'Normal'), (S_tri, 'Tri Modal')]:
            title_label = f'{label}'
            # Heatmap
            ax = sns.heatmap(S)
            ax.set(title=f'{label} S Matrix', **hmap_kwargs)
            save('s_matrix', label)

            # Distribution plots (ravel)
            ax = sns.displot(S.ravel())
            ax.set(title=f'{label} S Matrix Distribution', **dist_kwargs)
            save('s_dist', label, show=True)

            continue

            # Marginal Bar Plots
            marginal = np.sum(np.abs(S), axis=1)
            ax = sns.barplot(y=marginal, x=np.arange(marginal.shape[0]), **bar_kwargs)
            ax.set(title=f'{label} Marginal Variable Weight', **marg_kwargs)
            save('s_marginal', label, dpi=300)

            sorted_marginal = np.sort(marginal)
            ax = sns.barplot(y=marginal, x=np.arange(marginal.shape[0]), **bar_kwargs)
            ax.set(title=f'{label} Sorted Marginal Variable Weight', **marg_kwargs)
            save('s_sorted_marginal', label, dpi=300)
