from core.tasks import Cola, Calame
from core.sur_steps import surprisal_score_cola, surprisal_score_calame
from core.sur_steps import get_surprisal_scorer
import logging
import argparse

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Evaluation of Surprisal for linguistic datasets')
    parser.add_argument('--model', type=str, help='Model to use for obtaining text representations')
    parser.add_argument('--cache', type=str, help='Directory where cache data will be stored')
    parser.add_argument('--test', action='store_true', help='Test functionalities')
    parser.add_argument('--lang', type=str, help='Language of the dataset')
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate (CoLA or Calame)')
    parser.add_argument('--token', type=str, help='Hugging Face authentication token')
    args = parser.parse_args()

    if args.test:
        raise NotImplementedError
    
    model_scorer = get_surprisal_scorer(args.model, args.cache, args.token)
    if args.dataset == "calame":
        calame_task = Calame.Calame(args.lang, args.cache)
        calame_task.load_evaluation_dataset()
        surprisal_score_calame(model_scorer, args.model, calame_task.dataset)
    elif args.dataset == "cola":
        cola_task = Cola.Cola(args.lang, args.cache)
        cola_task.load_evaluation_dataset()
        surprisal_score_cola(model_scorer, args.model, cola_task.dataset_good, cola_task.dataset_bad)
    else:
        print("Dataset not suported...")
        exit()



    