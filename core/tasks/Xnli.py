from core.SurprisalTask import SurprisalTask
from datasets import load_dataset

class Xnli(SurprisalTask):
    def __init__(self, lang, cache_dir):
        self.lang = lang
        self.cache_dir = cache_dir
        self.dataset_contradiction = []
        self.dataset_entailment = []
        self.dataset_neutral = []

    def load_evaluation_dataset(self):
        if self.lang == "gl":
            self.dataset_contradiction, self.dataset_entailment, self.dataset_neutral = self.__load_gl()
        elif self.lang == "en":
            self.dataset_good, self.dataset_bad = self.__load_cola_en()
        elif self.lang == "cat":
            self.dataset_good, self.dataset_bad = self.__load_catcola()
        elif self.lang == "es":
            self.dataset_good, self.dataset_bad = self.__load_escola()
        else:
            print("XNLI language not suported...")
            exit()

    def __load_gl(self):
        dataset = load_dataset("proxectonos/xnli_gl", cache_dir = self.cache_dir)["test"]
        dataset_contradiction = [f"{item['sentence1']+item['sentence2']}" for item in dataset if item['gold_label'] == "contradiction"]
        dataset_entailment = [f"{item['sentence1']+item['sentence2']}" for item in dataset if item['gold_label'] == "entailment"]
        dataset_neutral = [f"{item['sentence1']+item['sentence2']}" for item in dataset if item['gold_label'] == "neutral"]
        return dataset_contradiction, dataset_entailment, dataset_neutral

    def __load_catcola(self):
        dataset = load_dataset("nbel/CatCoLA", cache_dir = self.cache_dir)["validation"]
        dataset_bad = [item['Sentence'] for item in dataset if item['Label'] == 0]
        dataset_good = [item['Sentence'] for item in dataset if item['Label'] == 1]
        return dataset_good, dataset_bad

    def __load_escola(self):
        dataset = load_dataset("nbel/EsCoLA", cache_dir = self.cache_dir)["validation"]
        dataset_bad = [item['Sentence'] for item in dataset if item['Label'] == 0]
        dataset_good = [item['Sentence'] for item in dataset if item['Label'] == 1]
        return dataset_good, dataset_bad

    def __load_cola_en(self):
        dataset = load_dataset("nyu-mll/glue", "cola", cache_dir = self.cache_dir)["validation"]
        dataset_bad = [item['sentence'] for item in dataset if item['label'] == 0]
        dataset_good = [item['sentence'] for item in dataset if item['label'] == 1]
        return dataset_good, dataset_bad