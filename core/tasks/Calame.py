from core.SurprisalTask import SurprisalTask
from datasets import load_dataset

class Calame(SurprisalTask):
    def __init__(self, lang, cache_dir):
        self.lang = lang
        self.cache_dir = cache_dir
        self.dataset = []
    
    def load_evaluation_dataset(self):
        if self.lang == "gl":
            self.__load_calame_gl()
        elif self.lang == "pt":
            self.__load_calame_pt()
        else:
            print("Calame language not suported...")
            exit()
    
    def __load_calame_pt(self):
        dataset = load_dataset("NOVA-vision-language/calame-pt", "all", cache_dir = self.cache_dir)["train"]
        dataset_concatenated = [item['sentence'] + ' ' + item['last_word'] for item in dataset]
        self.dataset = dataset_concatenated
    
    def __load_calame_gl(self):
        calame_gl_path = "/mnt/netapp1/Proxecto_NOS/adestramentos/avaliacion/eval_corpus/calame-gl/2025-01-22_calame_rev.json"
        dataset = load_dataset('json', data_files=calame_gl_path)["train"]
        dataset_concatenated = [item['target'] + ' ' + item['last_w_tgt'] for item in dataset]
        self.dataset = dataset_concatenated

