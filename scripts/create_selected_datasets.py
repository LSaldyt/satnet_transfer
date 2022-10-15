from satnet_transfer.download import *
import numpy as np
import satnet
from pathlib import Path, PurePath
from satnet_transfer.dataset import *
from functools import partial

base_url = 'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/'
ext = '.tar.gz'

selected_problems = [
    'PLANNING/BlocksWorld/blocksworld',
    'PLANNING/Logistics/logistics',
    'DIMACS/AIM/aim',
    'DIMACS/PARITY/parity',
    'DIMACS/HANOI/hanoi',
    'DIMACS/PHOLE/pigeon-hole',
    'Bejing/Bejing',
    'GCP/gcp-large',
    'QG/QG'
        ]
selected_problems = [(f'{base_url}{prob}{ext}', PurePath(f'{prob}{ext}'))
                     for prob in selected_problems]

def make_transforms(rng, size):
    transforms = [('mask', partial(mask, rng=rng, n_masks=size, n_samples=size, mask_sym=0))]
    return transforms

def run():
    cnf_dir  = Path('cnf/')
    cnf_dir.mkdir(exist_ok=True)
    data_dir = Path('data/')
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(2022)

    transforms = make_transforms(rng, size=10)

    for url, path in selected_problems:
        stem = path.with_suffix('').stem
        extract_dir = cnf_dir / Path(stem)
        extract_dir.mkdir(exist_ok=True)
        download_and_extract(url=url, target_path=path.name, extract_dir=extract_dir)
        try:
            for filename in extract_dir.glob(f'*.cnf'):
                print(filename)
                for transform_name, transform in transforms:
                    dataset_path = generate_dataset(filename, data_dir, rng,
                                                    transform=transform, limit=10000,
                                                    suffix=f'_{transform_name}.npz')
        except SolverFailedError:
            print(f'Solving failed for {stem}, skipping.')
