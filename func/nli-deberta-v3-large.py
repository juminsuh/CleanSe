import os
import pickle as pkl
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import settings
from func.metric import *

device = "cuda:0"

model_name = "cross-encoder/nli-deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
# print(model.config.id2label) # {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
print(f"model has been successfully loaded.")
def IntraScore(resultDict): 

    sequences = []
    # 각 sample의 generated_text를 clustering한 후 intrascore 계산
    for _, sample in tqdm(enumerate(resultDict), total=len(resultDict)):
        question = sample['question'] # 하나의 question 
        generated_texts = sample['generations'] # 10개의 generated texts 리스트
        last_embeddings = sample['last_embeddings'] # 10개의 generations에 대한 hidden states
        cosine_total = sample['cosine_total']
        # print(f"questison:", question): # 질문 하나 
        # print(f"generated_texts:", generated_texts) # 10개의 생성된 답변들
        # print(f"last_embeddings:", last_embeddings) # (10, 4096) 차원의 hidden states matrix
        # print(f"last_embeddings shape:", last_embeddings.shape) # (10, 4096)
        # print(f"cosine_total:", cosine_total) # total cosine similarity
        if last_embeddings is None: # None cannot be subscriptable
            intra_score = 0.0  
        else: 
            clusters = [[{'answer': generated_texts[0], 'embeddings': last_embeddings[0, :]}]]

            # clustering with nli model: 나머지 9개의 텍스트를 클러스터링
            for i in range(1, len(generated_texts)):

                already_add = False

                for c in clusters:
                    
                    qa_1 = question + ' ' + generated_texts[i] # 클러스터에 추가하고자 하는 answer
                    qa_2 = question + ' ' + c[0]['answer'] # 클러스터의 첫 번째 answer와만 비교 (의미의 transitive한 성질 이용)
                    
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

                    # print(qa_1, qa_2, predicted_label, reverse_predicted_label)

                    if predicted_label.item() == 1 and reverse_predicted_label.item() == 1: # 둘 다 entailment일 때만 같은 클러스터
                        c.append({'answer': generated_texts[i], 'embeddings': last_embeddings[i, :]})
                        already_add = True
                        break
                
                if not already_add:
                    clusters.append([{'answer': generated_texts[i], 'embeddings': last_embeddings[i]}])

            print("# of clusters:", len(clusters))
            # compute intrascore
            total_similarity = cosine_total
            intra_similarity = 0.0
            for c in clusters:
                num_cluster = len(c)
                if num_cluster < 2: # 2개 이하면 계산 불필요
                    continue 
                for i in range(num_cluster):
                    for j in range(i+1, num_cluster): # 같은 클러스터 내 서로 다른 임베딩 벡터끼리의 cosine similarity
                        intra_similarity += cosine_similarity_v2(c[i]['embeddings'], c[j]['embeddings'])
            print(f"total similarity: {total_similarity}")
            print(f"intra similarity: {intra_similarity}")
            intra_score = intra_similarity / total_similarity
            print("intra_score:", intra_score)

        cosine_score, _ = compute_CosineSimilarity_v2(last_embeddings)
        print(f"cosine_score:", cosine_score)

        curr_seq = dict(
            num_of_cluster = len(clusters),
            intra_score = intra_score
        )
        sequences.append(curr_seq)
    return sequences

if __name__ == "__main__":

    open_dir = 'mistral-7b_coqa_for_clustering'
    file_name = f"/mnt/aix7101/minsuh-output/{open_dir}/0.pkl"
    f = open(file_name, "rb")
    resultDict = pkl.load(f)

    dir = 'mistral-7b_coqa_nli-deberta-v3-large'
    cache_dir = os.path.join(settings.GENERATION_FOLDER, dir) # 최종 output 저장 경로
    os.makedirs(cache_dir, exist_ok=True) # 디렉토리 생성
    
    sequences = IntraScore(resultDict)
    pd.to_pickle(sequences, os.path.join(cache_dir, '0.pkl'))






        
            
                






