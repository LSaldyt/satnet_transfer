from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas  as pd

def run():
    sns.set_style('whitegrid')
    for metric in ('train_loss', 'test_loss', 'train_error', 'test_error'):
        for base in ('bw_large.b', 'bw_large.c', 'huge', 'medium', 'CBS_k3_n100_m403_b10_919', 'CBS_k3_n100_m403_b10_930', 'logistics.a', 'logistics.d', 'enddr2-10-by-5-8'):
            #metrics = pd.read_csv('metrics/CBS_k3_n100_m403_b10_999_masked.csv')
            path = Path(f'metrics/{base}_mask.csv')
            if path.is_file():
                metrics = pd.read_csv(path)
                sns_plot = sns.lineplot(metrics, y=metric, x='epoch', label=base)
        sns_plot.set(xlabel=f'Epoch', ylabel=f'{metric.title()}')
        plt.savefig(f'plots/{metric}.png')
        plt.show()
        plt.clf()
