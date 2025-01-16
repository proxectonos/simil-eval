from core.SurprisalTask import SurprisalTask
from datasets import load_dataset

class Calame(SurprisalTask):
    def __init__(self, lang, cache_dir):
        self.lang = lang
        self.cache_dir = cache_dir
        self.dataset = []
    
    def load_evaluation_dataset(self):
        dataset = load_dataset("NOVA-vision-language/calame-pt", "all", cache_dir = self.cache_dir)["train"]
        dataset_concatenated = [item['sentence'] + ' ' + item['last_word'] for item in dataset]
        self.dataset = dataset_concatenated

