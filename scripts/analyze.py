from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas  as pd

def run():
    for metric in ('train_loss', 'test_loss', 'train_error', 'test_error'):
        for base in ('bw_large.b', 'bw_large.c', 'huge', 'medium', 'CBS_k3_n100_m403_b10_919', 'CBS_k3_n100_m403_b10_930', 'logistics.a', 'logistics.d', 'enddr2-10-by-5-8'):
            #metrics = pd.read_csv('metrics/CBS_k3_n100_m403_b10_999_masked.csv')
            path = Path(f'metrics/{base}_mask.csv')
            if path.is_file():
                metrics = pd.read_csv(path)

                if 'error' in metric:
                    metrics[metric] = metrics[metric].apply(lambda s : float(s.split('(')[-1].split(')')[0]))
                sns_plot = sns.lineplot(metrics, y=metric, x='epoch', label=base)
                fig = sns_plot.get_figure()
        fig.savefig(f'{metric}.png')
        plt.show()
        plt.clf()
