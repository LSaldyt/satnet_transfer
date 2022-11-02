from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas  as pd

def run():
    metrics = pd.read_csv('metrics/basic.csv')
    sns.lineplot(metrics, y='test_error', x='epoch', palette='ch:r=-.2,d=.3_r')
    plt.show()
