from core.tasks import Belebele, Openbookqa, Veritasqa, Summarization
from core.sim_steps import EvaluatingModel
from core.sim_steps import create_examples, generate_completions, evaluate_similarity
import argparse
import logging

# Custom action to handle --token without a value
class OptionalString(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, values)

def test():
    print("Test function")
    return

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Evaluation of QA datasets using similarity')
    # General arguments
    parser.add_argument('--dataset', type=str, help='Dataset to evaluate')
    parser.add_argument('--model', type=str, help='Model to use for text generation')
    parser.add_argument('--cache', type=str, help='Directory where cache data will be stored')
    parser.add_argument('--token', type=str, nargs='?', default=None, action=OptionalString, help='Hugging Face authentication token')
    parser.add_argument('--test', action='store_true', help='Test functionalities')
    parser.add_argument('--language', type=str, help='Dataset language')
    # Example creation arguments
    parser.add_argument('--create_examples', action='store_true', help='Generate examples')
    parser.add_argument('--fewshot_num', type=int, help='Number of few-shots to generate (default is 5)')
    parser.add_argument('--examples_file', type=str, help='Name of the examples file')
    parser.add_argument('--show_options', type=lambda x: (str(x).lower() == 'true'), help='Include answer options when generating examples')
    # Generation arguments
    parser.add_argument('--generate_completions', action='store_true', help='generate_completions for the created examples')
    parser.add_argument('--results_file', type=str, help='Name of the results file')
    # Evaluation arguments
    parser.add_argument('--evaluate_similarity', action='store_true', help='Evaluate similarity between generated and original answers')
    parser.add_argument('--metrics', type=str, nargs='+', help='Metrics to use for similarity evaluation')

    args = parser.parse_args()
    print(args)

    if args.test:
        print("Test funcionalities")
        test()
        exit()

    if args.dataset == "belebele":
        task = Belebele.Belebele(lang=args.language, cache=args.cache)

    elif args.dataset == "openbookqa":
        task = Openbookqa.Openbookqa(lang=args.language, cache=args.cache, token=args.token)

    elif args.dataset == "veritasqa":
        task = Veritasqa.Veritasqa(lang=args.language, cache=args.cache)
    
    elif args.dataset == "summarization":
        task = Summarization.Summarization(lang=args.language, cache=args.cache)

    else:
        exit("Task not supported. Currently implemented tasks are [Belebele, OpenBookQA]")

    if args.create_examples:
        create_examples(task, examples_file=args.examples_file, fewshot_num=args.fewshot_num, show_options=args.show_options)
    
    evaluated_model = EvaluatingModel(args.model, args.cache, args.token)

    if args.generate_completions:
        generate_completions(task, evaluated_model, args.results_file, args.examples_file)
    
    if args.evaluate_similarity:
        supported_metrics = ["cosine","bertscore","moverscore"]
        for metric in args.metrics:
            if metric not in supported_metrics:
                exit(f"Unsupported metric: {metric}. Currently implemented metrics are {supported_metrics}")
        evaluate_similarity(task, evaluated_model, args.metrics, args.results_file)
    exit()