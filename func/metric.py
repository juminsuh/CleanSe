import numpy as np
from numpy.linalg import norm
import torch
import torch.nn.functional as F

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

def get_lenghthNormalized_entropy(batch_scores, num_tokens): # use multiple sequences
    seq_entropy = np.zeros(len(num_tokens))  
    for ind1, logits in enumerate(batch_scores): 
        for ind2, seq_logits in enumerate(logits):
            if ind1 < num_tokens[ind2]:
                conf, _ = torch.max(seq_logits.softmax(0), dim=0)
                seq_entropy[ind2] = seq_entropy[ind2] + np.log(conf.cpu().numpy())
    normalized_entropy = 0
    for ind, entropy in enumerate(seq_entropy):
        normalized_entropy += entropy/num_tokens[ind] # divide each sequence length
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
    for idx in range(hidden_states[1][-1].shape[0]): # 10 times
        embeddings[idx,:] = hidden_states[num_tokens[idx]-2][selected_layer][idx,0,:] # 마지막 토큰의 중간 레이어의 idx번째 sequence의 embedding_size, (10, 4096)
    return embeddings

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

def cosine_similarity_v2(vec1, vec2):
    vec1 = vec1.cpu().numpy()
    vec2 = vec2.cpu().numpy()
    return np.dot(vec1, vec2) / (norm(vec1)*norm(vec2))

def compute_CosineSimilarity(embeddings):
    if embeddings is None: # when both annotaions are empty in dataset
        return 0, 1e10
    concatenated_matrix = embeddings
    concatenated_matrix = concatenated_matrix.cpu().numpy().astype(float) 
    num_gens = concatenated_matrix.shape[0] # 10
    indices = list(range(num_gens))
    combinated = list(combinations(indices, 2)) # yield combination of indices
    cosine_similarities = []

    for i, j in combinated:
        similarity = cosine_similarity(concatenated_matrix[i], concatenated_matrix[j])
        cosine_similarities.append(similarity)
    
    total = sum(cosine_similarities)
    denominator = (num_gens*(num_gens-1))//2
    output = total/denominator

    return output, total

def compute_CosineSimilarity_v2(embeddings):
    if embeddings is None: # when both annotaions are empty in datase
        return 0, 1e10
    concatenated_matrix = embeddings
    num_gens = concatenated_matrix.shape[0] # 10
    indices = list(range(num_gens))
    combinated = list(combinations(indices, 2)) # yield combination of indices
    cosine_similarities = []

    for i, j in combinated:
        similarity = cosine_similarity_v2(concatenated_matrix[i], concatenated_matrix[j])
        cosine_similarities.append(similarity)
    
    total = sum(cosine_similarities)
    denominator = (num_gens*(num_gens-1))//2
    output = total/denominator

    return output, total


        
    
        




