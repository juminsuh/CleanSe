import datasets
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

def get_len_avg(data_name):
    if data_name == "SQuAD" or data_name == "coqa_dataset":
        open_dir = f"/mnt/aix7101/minsuh-dataset/{data_name}"
        data = datasets.load_from_disk(open_dir)
        lengths = [len(example['answer']['text']) for example in data]
        return np.mean(lengths)

if __name__ == '__main__':
    data_names = ["SQuAD", "coqa_dataset"]
    print(f"avg length of SQuAD: {get_len_avg(data_name=data_names[0]):.2f}")
    print(f"avg length of coqa: {get_len_avg(data_name=data_names[1]):.2f}")
