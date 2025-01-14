from core.SimilarityTask import SimilarityTask
from core.SimilarityTask import load_yaml, convert_to_dict
from datasets import load_dataset

class Summarization(SimilarityTask):
    """
    Class for the Summarization task.
    """
    
    def __init__(self, lang, cache):
        super().__init__(
            "summarization",
            lang,
            "RESPOST",
            cache)

    def get_correct_option(self, example): #Necessary to make minimal changes in eval_similarity code
        return self.get_correct_options(example)
    
    def get_options(self, example):
        correct_options = self.get_correct_options(example)
        incorrect_options = self.get_incorrect_options(example)
        options = correct_options + incorrect_options
        options = [option.strip() for option in options] #Remove leading and trailing whitespaces
        return options
    
    def get_best_answer(self, example):
        return example['best_answer']
    
    def build_prompt(self,example, show_answer, show_options = True):

        prompt = ""
        options = self.get_options(example)
        if show_options:
            if show_answer:
                correct_option = self.get_correct_option(example)
                prompt = f"""PREGUNTA: {example['question_stem']} 
                OPCION A: {options[0]}
                OPCION B: {options[1]}
                OPCION C: {options[2]}
                OPCION D: {options[3]}
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""PREGUNTA: {example['question_stem']} 
                OPCION A: {options[0]}
                OPCION B: {options[1]}
                OPCION C: {options[2]}
                OPCION D: {options[3]}
                RESPOST"""
        else:
            if show_answer:
                correct_option = self.get_correct_option(example)
                prompt = f"""PREGUNTA: {example['question_stem']} 
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""PREGUNTA: {example['question_stem']} 
                RESPOST"""
        return prompt
    
    def load_data(self):
        data_yaml = load_yaml()
        veritas_dict = convert_to_dict(data_yaml['veritasqa'])
        hf_dataset = veritas_dict[self.lang]
        print(f"DATASET: {hf_dataset}")
        if hf_dataset=="None":
            exit(f"Dataset not found for {self.name} and language {self.lang}")
        repo=hf_dataset[0]
        lang_subset=hf_dataset[1]
        print(f"DATASET: {repo} - {lang_subset}")
        self.dataset = load_dataset(repo, cache_dir = self.cache)[lang_subset]
        return self.dataset