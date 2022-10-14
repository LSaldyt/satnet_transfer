from satnet_transfer.download import *

def run():
    # download_and_extract()
    download_and_extract(url='https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m403_b10.tar.gz',
            target_path='CBS_k3_n100_m403_b10.tar.gz',
            extract_dir='cnf')
