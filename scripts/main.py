import numpy as np
import satnet
from time import time

def run():
    threshold = 1000
    rng = np.random.default_rng(2022)
    s   = Settings(batch_size=32, lr=2e-3, epochs=10, split=0.8)

    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)

    data_dir = Path('data/')
    for problem_file in data_dir.glob('*.npz'):
        stem = problem_file.stem
        s.update(metrics_file=f'metrics/{stem}.csv')

        dataset = SATDataset(problem_file)
        print(f'Considering problem {stem}')
        if len(dataset) < threshold:
            print(f'The dataset in {problem_file} is too small ({len(dataset)})')
            continue
        print(f'Optimizing {stem} ({len(dataset)} examples)')

        inp, inp_mask, lbl = dataset[0]
        n   = inp.shape[0]
        m   = 1000
        aux = 1000
        sat = satnet.SATNet(n, m, aux)

        optimizer = torch.optim.AdamW(sat.parameters(), lr=s.lr)
        loop(sat, dataset, optimizer, s)
    # # filename = 'hanoi5.cnf'
    # filename = 'CBS_k3_n100_m403_b10_999.cnf'
    # start = time()
    # count = 0
    # for example in generate_from(filename, rng, n_masks=100, n_samples=100):
    #     print(example)
    #     count += 1
    # end = time()
    # print(f'Found {count} examples in {end - start} seconds')
