import os
from tqdm import tqdm

dcts = ["quadratic"]
seeds = [325, 326, 327, 328, 329, 330, 331, 332, 333, 334]
sample_sizes = [2, 4, 6, 8, 12, 16]

print("dct,num_samples,seed,vecind,score")
for dct in dcts:
    for seed in seeds:
        for num_samples in tqdm(sample_sizes):
            os.system(f"python3 eval_harmful.py {dct} {seed} {num_samples}")
