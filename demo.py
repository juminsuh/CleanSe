import os
import pickle as pkl
import sys

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import settings
from func.metric import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-7b-hf')
parser.add_argument('--dataset', type=str, default='SQuAD')
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--project_ind', type=str, default='for_clustering')
parser.add_argument('--clustering', type=str, default='demo')

args = parser.parse_args()
logInfo = open(f"{settings.GENERATION_FOLDER}/logInfo_{args.model}_{args.dataset}_demo.txt", mode="w",encoding="utf-8")

def load_clustering_model(model_name, device):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    print(f"Model has been successfully loaded.")
    
    return tokenizer, model

def CleanSeScore(resultDict, tokenizer, model, args, threshold): 

    device = args.device
    for i, sample in tqdm(enumerate(resultDict), total=len(resultDict)):
        if i > 50:
            break
        question = sample['question'] # one question
        answer = sample['answer']
        response = sample['most_likely_generation'] # most likely answer
        generated_texts = sample['generations'] # 10 generations
        last_embeddings = sample['last_embeddings'] # 10 last embeddings: (10, 4096)
        cosine_total = sample['cosine_total']

        if last_embeddings is None: # None cannot be subscriptable
            cleanse_score = 0.0  
        else: 
            clusters = [[{'answer': generated_texts[0], 'embeddings': last_embeddings[0, :]}]]

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
                        c.append({'answer': generated_texts[i], 'embeddings': last_embeddings[i, :]})
                        already_add = True
                        break
                
                if not already_add:
                    clusters.append([{'answer': generated_texts[i], 'embeddings': last_embeddings[i]}]) # create new cluster
                
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
            
            cleanse_score = intra_similarity / total_similarity
            print("cleanse_score:", cleanse_score)
        
        print("Question:", question, file=logInfo)
        print("GTAnswer:", answer, file=logInfo)
        print("Response:", response, file=logInfo)
        criteria = f"{args.model}-{args.dataset}"
        if cleanse_score < threshold[criteria]:
            print("Decision:", '''⚠️ Note: This response may contain generated content that is not factually accurate. 
                  Please verify the information before using it in critical contexts.''', file=logInfo)
        else:
            print("Decision:", '''✅ This seems to be a reliable and accurate response.''', file=logInfo)
        print("\n","\n","\n", file=logInfo)

if __name__ == "__main__":

    open_dir = f'{args.model}_{args.dataset}_{args.project_ind}'
    file_name = f"{settings.GENERATION_FOLDER}/{open_dir}/0.pkl"
    f = open(file_name, "rb")
    resultDict = pkl.load(f)

    dir = f'{args.model}_{args.dataset}_{args.clustering}'
    cache_dir = os.path.join(settings.GENERATION_FOLDER, dir) 
    os.makedirs(cache_dir, exist_ok=True) # make directory
    
    threshold = {'llama-7b-hf-SQuAD': 0.5786, 'llama-7b-hf-coqa': 0.5357,
                'llama-13b-hf-SQuAD': 0.5446, 'llama-13b-hf-coqa': 0.5033,
                'Llama-2-7b-hf-SQuAD': 0.5369, 'Llama-2-7b-hf-coqa': 0.4858,
                'mistral-7b-SQuAD': 0.8585, 'mistral-7b-coqa': 0.6473}
    
    model_name = "cross-encoder/nli-deberta-v3-base"
    tokenizer, model = load_clustering_model(model_name, args.device)
    CleanSeScore(resultDict, tokenizer, model, args, threshold)
