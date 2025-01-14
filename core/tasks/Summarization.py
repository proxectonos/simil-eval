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
        return example["summary"]
    
    def get_options(self, example):
        raise NotImplementedError("This method is not implemented for this task")

    def build_prompt(self,example, show_answer, show_options = True):

        prompt = ""
        if show_options:
            if show_answer:
                correct_option = self.get_correct_option(example)
                prompt = f"""TEXTO: {example['text']} 
                RESUMO: {correct_option}\n"""
            else:
                prompt = f"""TEXTO: {example['text']} 
                RESUM"""
        else:
            raise NotImplementedError("No show options not has sense for this task")
        return prompt
    
    def load_data(self):
        data_yaml = load_yaml()
        veritas_dict = convert_to_dict(data_yaml['summarization'])
        dataset_path = veritas_dict[self.lang]
        self.dataset = load_dataset('json', data_files=dataset_path)
        print(self.dataset)
        return self.dataset