python -m pipeline.generate --model Llama-2-7b-hf --dataset SQuAD --fraction_of_data_to_use 1 --project_ind for_clustering
python -m clustering.nli-deberta-v3-base --model Llama-2-7b-hf --dataset SQuAD --project_ind for_clustering --clustering nli-deberta-v3-base
