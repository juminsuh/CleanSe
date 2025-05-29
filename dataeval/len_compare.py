import datasets
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

def get_len_distribution(data_name):
    if data_name == "SQuAD" or data_name == "coqa_dataset":
        open_dir = f"/mnt/aix7101/minsuh-dataset/{data_name}"
        data = datasets.load_from_disk(open_dir)
        return [len(example['answer']['text']) for example in data]
    elif data_name == "nq":
        data = load_dataset("nq_open", split="validation")
        all_answers = [example.pop('answer') for example in data]
        return [len(sample[0]) for sample in all_answers]

def plot_len_distribution(data_names):

    data_lengths = [get_len_distribution(data_name) for data_name in data_names]

    plt.figure(figsize=(10, 5))
    plt.boxplot(data_lengths, tick_labels=data_names, patch_artist=True, showmeans=True)

    plt.title('Box Plot of Answer Lengths')
    plt.xlabel('Dataset')
    plt.ylabel('Answer Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.savefig("/mnt/aix7101/minsuh-output/Len_Dist_Boxplot.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    data_names = ["SQuAD", "coqa_dataset", "nq"]
    plot_len_distribution(data_names=data_names)
