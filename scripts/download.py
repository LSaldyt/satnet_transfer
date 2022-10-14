import requests
import tarfile
from pathlib import Path

def download_and_extract(
    url='https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/DIMACS/HANOI/hanoi.tar.gz',
    target_path='hanoi.tar.gz',
    extract_dir='cnf'):

    target_path = Path(target_path)
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())

        if tarfile.is_tarfile(target_path):
            with tarfile.open(target_path) as f:
                f.extractall(path=extract_dir)
    finally:
        target_path.unlink()

def run():
    # download_and_extract()
    download_and_extract(url='https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m403_b10.tar.gz',
            target_path='CBS_k3_n100_m403_b10.tar.gz',
            extract_dir='cnf')
