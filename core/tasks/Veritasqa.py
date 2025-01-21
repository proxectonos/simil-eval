from core.SimilarityTask import SimilarityTask
from core.SimilarityTask import load_yaml, convert_to_dict
from datasets import load_dataset
import random

class Veritasqa(SimilarityTask):
    """
    Class for the VeritasQA task.
    """
    
    def __init__(self, lang, cache):
        super().__init__(
            "veritasqa",
            lang,
            "RESPOST",
            cache)
    
    def get_best_answer(self, example):
        return example["best_answer"]
    
    def get_correct_options(self, example):
        if ";" in example["correct_answers"]:
            return example["correct_answers"].split(";")
        return [example["correct_answers"]]

    def get_incorrect_options(self, example):
        if ";" in example["incorrect_answers"]:
            return example["incorrect_answers"].split(";")
        return [example["incorrect_answers"]]

    def get_correct_option(self, example): #Necessary to make minimal changes in eval_similarity code
        return self.get_correct_options(example)
    
    def get_options(self, example):
        correct_options = self.get_correct_options(example)
        incorrect_options = self.get_incorrect_options(example)
        options = correct_options + incorrect_options
        options = [option.strip() for option in options] #Remove leading and trailing whitespaces
        random.shuffle(options)
        return options
    
    def build_prompt(self,example, show_answer, show_options):

        prompt = ""
        options = self.get_options(example)
        if show_options:
            formated_options = "\n".join([f"OPCION {i+1}: {option}" for i, option in enumerate(options)])
            if show_answer:
                correct_option = self.get_best_answer(example)
                prompt = f"""   PREGUNTA: {example['question']}\n{formated_options}\nRESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""   PREGUNTA: {example['question']}\n{formated_options}\nRESPOST"""
        else:
            if show_answer:
                correct_option = self.get_best_answer(example)
                prompt = f"""CONTEXTO: {example['question']} 
                PREGUNTA: {example['question']}
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""CONTEXTO: {example['question']} 
                PREGUNTA: {example['question']}
                RESPOST"""
        return prompt
    
    def load_data(self):
        data_yaml = load_yaml()
        data_dict = convert_to_dict(data_yaml['veritasqa'])
        hf_dataset = data_dict[self.lang]
        print(f"Dataset: {hf_dataset}")
        if hf_dataset=="None":
            exit(f"Dataset not found for {self.name} and language {self.lang}")
        repo=hf_dataset[0]
        lang_subset=hf_dataset[1]
        print(f"Dataset: {repo} - {lang_subset}")
        self.dataset = load_dataset(repo, cache_dir = self.cache)[lang_subset]
        return self.dataset