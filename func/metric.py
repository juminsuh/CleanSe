import numpy as np
from numpy.linalg import norm
import torch

from rouge_score import rouge_scorer
from sentence_transformers import util
from itertools import combinations

rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def getRouge(rouge, generations, answers):
    # results = rouge.compute(predictions=[generations], references=[answers], use_aggregator=False)
    results = rouge.score(target = answers, prediction = generations)
    RoughL = results["rougeL"].fmeasure  #fmeasure/recall/precision
    return RoughL

def get_perplexity_score(scores):
    perplexity = 0.0
    for logits in scores:
        conf = torch.max(logits.softmax(1)).cpu().item()
        perplexity += np.log(conf)
    perplexity = -1.0 * perplexity/len(scores)
    return perplexity

### batch_scores ([[logits]], [[logits]], [[logits]])
### num_tokens : list 
def get_entropy_score(batch_scores, num_tokens):  
    Conf = []
    for logits in batch_scores:
        conf, index = torch.max(logits.softmax(1), dim=1)
        Conf.append(conf.cpu().numpy())
    Conf = np.array(Conf)  
    Conf = Conf + 1e-6
    entropy = -1.0 * np.sum(np.log(Conf))/logits.shape[0]
    return entropy

def get_lenghthNormalized_entropy(batch_scores, num_tokens): # 여러 개의 batch (시퀀스) 이용
    seq_entropy = np.zeros(len(num_tokens))  
    for ind1, logits in enumerate(batch_scores): 
        for ind2, seq_logits in enumerate(logits):
            if ind1 < num_tokens[ind2]:
                conf, _ = torch.max(seq_logits.softmax(0), dim=0)
                seq_entropy[ind2] = seq_entropy[ind2] + np.log(conf.cpu().numpy())
    normalized_entropy = 0
    for ind, entropy in enumerate(seq_entropy):
        normalized_entropy += entropy/num_tokens[ind] # 각 시퀀스 길이로 나눔
    normalized_entropy = -1.0* normalized_entropy/len(num_tokens) 
    return normalized_entropy

def getLexicalSim(generated_texts):
    LexicalSim = 0
    for i in range(len(generated_texts)):
        for j in range(len(generated_texts)):
            if j<=i:
                continue
            LexicalSim += getRouge(rougeEvaluator, generated_texts[i], generated_texts[j])
    LexicalSim = LexicalSim/(len(generated_texts)*(len(generated_texts)-1)/2)
    return LexicalSim

def extract_embeddings(hidden_states, num_tokens):
    '''
    num_tokens[idx]-2: [EOS] 토큰 이전 토큰을 의미. 즉, 마지막 토큰
    selected_layer: 중간층
    idx: num_return_sequences
    [0, :]: 4096 차원의 임베딩
    추출해서 concatenated_matrix의 [idx, :]에 저장
    '''
    selected_layer = int(len(hidden_states[0])/2) # the number of layers = int(33/2) = 16

    if len(hidden_states) < 2:
        return None
    embeddings = torch.zeros(hidden_states[1][-1].shape[0], hidden_states[1][-1].shape[2]).to("cuda:0") # (10, 4096)
    for idx in range(hidden_states[1][-1].shape[0]): # 10번
        embeddings[idx,:] = hidden_states[num_tokens[idx]-2][selected_layer][idx,0,:] # 마지막 토큰의 중간 레이어의 idx번째 sequence의 embedding_size, (10, 4096)
    return embeddings

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

def cosine_similarity_v2(vec1, vec2):
    vec1 = vec1.cpu().numpy()
    vec2 = vec2.cpu().numpy()
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

def compute_CosineSimilarity(embeddings):
    if embeddings is None: # when both annotaions are empty in nq
        return 0, 1e10
    concatenated_matrix = embeddings
    concatenated_matrix = concatenated_matrix.cpu().numpy().astype(float) # cpu로 보냄, float64
    num_gens = concatenated_matrix.shape[0] # 10
    indices = list(range(num_gens))
    combinated = list(combinations(indices, 2)) # indices들을 2개씩 조합을 만듦
    cosine_similarities = []

    for i, j in combinated:
        similarity = cosine_similarity(concatenated_matrix[i], concatenated_matrix[j])
        cosine_similarities.append(similarity)
    
    total = sum(cosine_similarities)
    denominator = (num_gens*(num_gens-1))//2
    output = total/denominator

    return output, total

def compute_CosineSimilarity_v2(embeddings):
    if embeddings is None: # when both annotaions are empty in nq
        return 0, 1e10
    concatenated_matrix = embeddings
    num_gens = concatenated_matrix.shape[0] # 10
    indices = list(range(num_gens))
    combinated = list(combinations(indices, 2)) # indices들을 2개씩 조합을 만듦
    cosine_similarities = []

    for i, j in combinated:
        similarity = cosine_similarity_v2(concatenated_matrix[i], concatenated_matrix[j])
        cosine_similarities.append(similarity)
    
    total = sum(cosine_similarities)
    denominator = (num_gens*(num_gens-1))//2
    output = total/denominator

    return output, total



    
# def update_centroid(cluster):
#     return np.mean(cluster, axis = 0)

# cosine similarity >= threshold를 기준으로 k개의 generation을 clustering
# def clustering(hidden_states, num_tokens, threshold = 0.9):
#     last_embeddings = extract_embeddings(hidden_states, num_tokens)
#     last_embeddings = last_embeddings.cpu().numpy().astype(float) # float64
#     num_gens = last_embeddings.shape[0] # 10
#     clusters = [{'centroid': last_embeddings[0], 'element': [last_embeddings[0]]}] # 전체 cluster를 담아둘 리스트. 첫 번째 임베딩으로 initialize
#     '''
#     각 클러스터를 딕셔너리로 정의함. 
#     임베딩의 cosine similarity를 centroid와 계산하여 직접 하나의 요소와 비교했을 때보다 민감도를 낮춤. 
#     ablation study 1) threshold 조절 
#     ablation study 2) 직접 비교 vs centroid를 두고 비교
#     ''' 

#     for i in range(1, num_gens): # 첫 번째 임베딩을 제외한 모든 나머지 임베딩에 대해서
#         already_add = False
#         for c in clusters: # 각 distinct cluster와의 cosine similarity를 비교
#             similarity = cosine_similarity(last_embeddings[i], c['centroid']) # 각 distinct cluster의 평균과 비교
#             if similarity >= threshold: # threshold: 이 부분은 나중에 조정 (매우 민감할 수도)
#                 c['element'].append(last_embeddings[i]) # cos_sim이 threshold 이상이면 같은 클러스터로 append
#                 c['centroid'] = update_centroid(c['element'])
#                 already_add = True
#                 break
#         if not already_add:
#             clusters.append({'centroid': last_embeddings[i], 'element': [last_embeddings[i]]}) # add distinct cluster
    
#     return clusters

# cosine similarity >= threshold를 기준으로 k개의 generation을 clustering
# def clustering(hidden_states, num_tokens, threshold = 0.9):
#     last_embeddings = extract_embeddings(hidden_states, num_tokens)
#     last_embeddings = last_embeddings.cpu().numpy().astype(float) # float64
#     num_gens = last_embeddings.shape[0] # 10
#     clusters = [[last_embeddings[0]]] # 전체 cluster를 담아둘 리스트. 첫 번째 임베딩으로 initialize

#     for i in range(1, num_gens): # 첫 번째 임베딩을 제외한 모든 나머지 임베딩에 대해서
#         already_add = False
#         for c in clusters: # 각 distinct cluster와의 cosine similarity를 비교
#             similarity = cosine_similarity(last_embeddings[i], c[0]) # 각 distinct cluster에서 하나의 element랑만 비교
#             if similarity >= threshold: # threshold: 이 부분은 나중에 조정 (매우 민감할 수도)
#                 c.append(last_embeddings[i]) # cos_sim이 threshold 이상이면 같은 클러스터로 append
#                 already_add = True
#                 break
#         if not already_add:
#             clusters.append([last_embeddings[i]]) # add distinct cluster
#     return clusters
            
# # compute intrascore
# def compute_IntraScore(total, clusters):
#     total_similarity = total
#     intra_similarity = 0.0
#     for c in clusters:
#         num_cluster = len(c)
#         if num_cluster < 2: # 2개 이하면 계산 불필요
#             continue 
#         for i in range(num_cluster):
#             for j in range(i+1, num_cluster): # 같은 클러스터 내 서로 다른 임베딩 벡터끼리의 cosine similarity
#                 intra_similarity += cosine_similarity(c[i], c[j])
#     print(f"total similarity: {total_similarity}")
#     print(f"intra similarity: {intra_similarity}")
#     intra_score = intra_similarity / total_similarity
#     return intra_score

        
    
        




