from utils.surprisal import get_surprisal, get_surprisal_last_word
from utils.surprisal import get_scorer
from utils.metrics import difsur_score
import numpy as np

def get_surprisal_scorer(model_name, cache_dir, tokenHF):
    model_scorer = get_scorer(model_name, cache_dir, tokenHF)
    return model_scorer

def surprisal_score_calame(model_scorer, model_name, dataset):
    """Compute surprisal scores for the last word in a sentence for Calame dataset"""
    scores = []
    for examples in dataset:
        score = get_surprisal_last_word(model_scorer, examples)
        scores.append(score)

    scores_mean = round(np.mean([x for x in scores if x != None]),4)
    print(f"""Results for model: {model_name}
Mean score last word: {scores_mean}
{"#"*40}""")

def surprisal_score_difsur(model_scorer, model_name, dataset_good, dataset_bad, dataset_name):
    """Compute difsur scores using good and bad partitions of a dataset"""
    good_scores = []
    bad_scores = []
    for good in dataset_good:
        good_score = get_surprisal(model_scorer, good)
        good_scores.append(good_score)
    
    for bad in dataset_bad:
        bad_score = get_surprisal(model_scorer, bad)
        bad_scores.append(bad_score)

    good_mean = round(np.mean(good_scores),4)
    bad_mean = round(np.mean(bad_scores),4)
    difference_mean = round(bad_mean - good_mean,4)
    difsur = round(difsur_score(good_mean, bad_mean),4)
    print(f"""Results for model: {model_name}
Good mean: {good_mean}
Bad mean: {bad_mean}
difsur-{dataset_name}: {difsur}
{"#"*40}""")

