from core.SimilarityTask import SimilarityTask
from core.SimilarityTask import load_yaml, convert_to_dict
from datasets import load_dataset

class Belebele(SimilarityTask):
    """
    Class for the Belebele task.
    """
    def __init__(self, lang, cache):
        super().__init__(
            "belebele",
            lang,
            "RESPOST",
            cache)
        
    def get_correct_option(self, example):

        answers_dict = {"1":"mc_answer1",
                "2":"mc_answer2",
                "3":"mc_answer3",
                "4":"mc_answer4"}

        return example[answers_dict[example["correct_answer_num"]]]

    def get_options(self, example):

        return [example["mc_answer1"], example["mc_answer2"], example["mc_answer3"], example["mc_answer4"]]

    def build_prompt(self,example, show_answer, show_options):

        prompt = ""
        options = self.get_options(example)
        if show_options:
            if show_answer:
                correct_option = self.get_correct_option(example)
                prompt = f"""CONTEXTO: {example['flores_passage']} 
                PREGUNTA: {example['question']} 
                OPCION 1: {options[0]}
                OPCION 2: {options[1]}
                OPCION 3: {options[2]}
                OPCION 4: {options[3]}
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""CONTEXTO: {example['flores_passage']} 
                PREGUNTA: {example['question']} 
                OPCION 1: {options[0]}
                OPCION 2: {options[1]}
                OPCION 3: {options[2]}
                OPCION 4: {options[3]}
                RESPOST"""
        else:
            if show_answer:
                correct_option = self.get_correct_option(example)
                prompt = f"""CONTEXTO: {example['flores_passage']} 
                PREGUNTA: {example['question']}
                RESPOSTA: {correct_option}\n"""
            else:
                prompt = f"""CONTEXTO: {example['flores_passage']} 
                PREGUNTA: {example['question']}
                RESPOST"""
        return prompt

    def load_data(self):
        data_yaml = load_yaml()
        belebele_dict = convert_to_dict(data_yaml['belebele'])
        hf_dataset = belebele_dict[self.lang]
        print(f"DATASET: {hf_dataset}")
        if hf_dataset=="None":
            exit(f"Dataset not found for {self.name} and language {self.lang}")
        elif isinstance(hf_dataset, list): #Belebele subsets from original Belebele HF repo
            repo=hf_dataset[0]
            subset=hf_dataset[1]
            print(f"DATASET: {repo} - {subset}")
            self.dataset = load_dataset(repo, subset, cache_dir = self.cache)["test"]
        else: #Galician belebele
            self.dataset = load_dataset(hf_dataset, cache_dir = self.cache)["train"]
        print("DATASET CARGADO!")
        return self.dataset