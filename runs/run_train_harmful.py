import os 
from tqdm import tqdm
dcts = ["linear"]
seeds = [325,326,327,328,329,330,331,332,333,334]
sample_sizes = [2,4,6,8,12,16]

for dct in dcts:
    for seed in seeds:
        print(dct, seed)
        for num_samples in tqdm(sample_sizes):
            os.system(f"python3 train_harmful.py {dct} {seed} {num_samples}")  
