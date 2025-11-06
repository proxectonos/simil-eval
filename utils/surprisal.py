import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from minicons import scorer

def get_scorer(model_name, cache_dir, tokenHF):
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, return_dict=True, token=tokenHF)
    model_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir , use_fast=True, token=tokenHF)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_scorer = scorer.IncrementalLMScorer(model, tokenizer=model_tokenizer, device=device)
    return model_scorer

def get_surprisal(model_scorer, sentence):
    return model_scorer.sequence_score(sentence, reduction = lambda x: -x.sum(0).item())

def normalize_token(tok: str) -> str:
    return tok.replace("▁", "")

def get_surprisal_last_word(model_scorer, string):
    words = string.split()
    last_word = words[-1]
    surprisals = model_scorer.token_score(string, surprisal=True)[0]
    last_surprisal_word = normalize_token(surprisals[-1][0])

    # Check if the last word is the same as the last surprisal word
    if last_word == last_surprisal_word:
        return surprisals[-1][1]
    # If not, concatenate previous elements until they match
    for i in range(2, len(words) + 1):
        last_surprisal_word = ''.join(normalize_token(word) for word, _ in surprisals[-i:])
        #word, score = surprisals[-i]; print(f"    {word} {score}")
        if last_word == last_surprisal_word:
            return max([score for _, score in surprisals[-i:]])
     return None