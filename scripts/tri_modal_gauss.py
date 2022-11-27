# from satnet.models import tri_modal_gauss
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas  as pd
import numpy as np
import math

def gauss(μ, σ):
    return (np.sign(μ) *
            np.exp(-(μ**2)/(2*(σ**2)))
            / (σ * np.sqrt(2. * math.pi)))

def tri_modal_gauss(x, σ):
    return gauss(x + 1, σ) + gauss(x, σ) + gauss(x - 1, σ)

def run():
    sns.set_style('whitegrid')

    r = 1000
    base = np.linspace(-1, 1, r)

    ax = sns.histplot(base)

    plt.show()

    # for dev in [0.125, 0.15, 0.2, 0.25]: #, 0.4, 0.5, 1.0, 2.0]::
    # for dev in [0.2, 0.15, 0.1]:
    for dev in np.linspace(0.2, 0.1, 10):
        sparse = tri_modal_gauss(base, dev)
        print(sparse)
        print(dev, np.trapz(np.abs(sparse)) / r)
        ax = sns.histplot(sparse)
        ax.set(title=f'Tri-Modal with Stdev {dev}')
        plt.show()
