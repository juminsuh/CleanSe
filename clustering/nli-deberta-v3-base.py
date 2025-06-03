import os
import pickle as pkl
import sys

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import settings
from func.metric import *

device = "cuda:0"

model_name = "cross-encoder/nli-deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
print(f"model has been successfully loaded.")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-7b-hf')
parser.add_argument('--dataset', type=str, default='coqa')
parser.add_argument('--project_ind', type=str, default='for_clustering')
parser.add_argument('--clustering', type=str, default='done_clustering')

args = parser.parse_args()

def CleanSeScore(resultDict): 

    sequences = []
    for _, sample in tqdm(enumerate(resultDict), total=len(resultDict)):
        question = sample['question'] # one question
        generated_texts = sample['generations'] # 10 generations
        seq_log_probs = sample['seq_log_probs'] # 10 seq_probs (10, )
        # print(f"seq_log_probs: {seq_log_probs}")
        last_embeddings = sample['last_embeddings'] # 10 last embeddings: (10, 4096)
        cosine_total = sample['cosine_total']

        if last_embeddings is None: # None cannot be subscriptable
            cleanse_score = 0.0  
        else: 
            clusters = [[{'answer': generated_texts[0], 'embeddings': last_embeddings[0, :], 'probability': seq_log_probs[0]}]]

            # clustering with nli model
            for i in range(1, len(generated_texts)):

                already_add = False
                for c in clusters:
                    qa_1 = question + ' ' + generated_texts[i] # answer which should be included to a certain cluster
                    qa_2 = question + ' ' + c[0]['answer'] # first member of cluster c
                    
                    inputs = tokenizer(qa_1, qa_2, return_tensors="pt", padding=True, truncation=True)
                    inputs = {key:value.to(device) for key, value in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        logits = outputs.logits
                    
                    predicted_label = torch.argmax(logits, dim=1)
                    
                    reverse_inputs = tokenizer(qa_2, qa_1, return_tensors="pt", padding=True, truncation=True)
                    reverse_inputs = {key:value.to(device) for key, value in reverse_inputs.items()}
                    
                    with torch.no_grad():
                        reverse_outputs = model(**reverse_inputs)
                        reverse_logits = reverse_outputs.logits
                    
                    reverse_predicted_label = torch.argmax(reverse_logits, dim=1)

                    if predicted_label.item() == 1 and reverse_predicted_label.item() == 1: # if bi-directional entailment then add 
                        c.append({'answer': generated_texts[i], 'embeddings': last_embeddings[i, :], 'probability': seq_log_probs[i]})
                        already_add = True
                        break
                
                if not already_add:
                    clusters.append([{'answer': generated_texts[i], 'embeddings': last_embeddings[i], 'probability': seq_log_probs[i]}]) # create new cluster

            print("# of clusters:", len(clusters))
                
            # compute cleanse score
            total_similarity = cosine_total
            intra_similarity = 0.0
            for c in clusters:
                num_cluster = len(c)
                if num_cluster < 2:
                    continue 
                for i in range(num_cluster):
                    for j in range(i+1, num_cluster): # cosine similarity between outputs within the same cluster
                        intra_similarity += cosine_similarity_v2(c[i]['embeddings'], c[j]['embeddings'])
            
            print(f"total similarity: {total_similarity}")
            print(f"intra similarity: {intra_similarity}")
            cleanse_score = intra_similarity / total_similarity
            print("cleanse_score:", cleanse_score)

        cosine_score, _ = compute_CosineSimilarity_v2(last_embeddings)
        print(f"cosine_score:", cosine_score)

        curr_seq = dict(            
            num_of_cluster = len(clusters),
            cleanse_score = cleanse_score
        )
        sequences.append(curr_seq)
    return sequences

if __name__ == "__main__":

    open_dir = f'{args.model}_{args.dataset}_{args.project_ind}'
    file_name = f"{settings.GENERATION_FOLDER}/{open_dir}/0.pkl"
    f = open(file_name, "rb")
    resultDict = pkl.load(f)

    dir = f'{args.model}_{args.dataset}_{args.clustering}'
    cache_dir = os.path.join(settings.GENERATION_FOLDER, dir) 
    os.makedirs(cache_dir, exist_ok=True) # make directory
    
    sequences = CleanSeScore(resultDict)
    pd.to_pickle(sequences, os.path.join(cache_dir, '0.pkl'))






        
            
                






