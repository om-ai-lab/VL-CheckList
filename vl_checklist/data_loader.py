import yaml
import os
import json

IMG_DIR = "/network/projects/aishwarya_lab/datasets/"

class DataLoader(object):
    def __init__(self, corpus_names, type ,task='itm', version="v1") -> None:
        self.root_dir = os.path.dirname(os.path.realpath(__file__))#.replace("/vl_checklist", "")
        self.cur_dir = os.path.realpath(os.curdir)

        self.version = version
        if task == 'itm':
            self.data = self.load_pos_and_neg_samples(corpus_names,type)
        elif task == 'itc':
            self.data = self.load_itc_samples(corpus_names,type)


    def load_pos_and_neg_samples(self, corpus_names: list,type):
        corpus = {}
        for corpus_name in corpus_names:
            corpus[corpus_name] = []
            config = yaml.load(open(os.path.join(self.cur_dir, 'corpus',  self.version, type, f'{corpus_name}.yaml'), 'r'), Loader=yaml.FullLoader)
            print(config["ANNO_PATH"])
            m = json.load(open(config["ANNO_PATH"]))
            for x in m:
                path, texts_dict = x
                path = os.path.join(IMG_DIR, config["IMG_ROOT"], path)
                if os.path.exists(path):
                    corpus[corpus_name].append({
                        "path": path,
                        "texts": texts_dict["POS"],
                        "label": 1
                    })
                    corpus[corpus_name].append({
                        "path": path,
                        "texts": texts_dict["NEG"],
                        "label": 0
                    })
        return corpus

    def load_itc_samples(self, corpus_names: list,type):
        corpus = {}
        for corpus_name in corpus_names:
            corpus[corpus_name] = []
            config = yaml.load(open(os.path.join(self.cur_dir, 'corpus',  self.version, type,f'{corpus_name}.yaml'), 'r'), Loader=yaml.FullLoader)
            m = json.load(open(os.path.join(self.cur_dir,config["ANNO_PATH"])))
            for x in m:
                path, texts_dict = x
                path = os.path.join(IMG_DIR, config["IMG_ROOT"], path)
                if os.path.exists(path):
                    corpus[corpus_name].append({
                        "path": path,
                        "texts_pos": texts_dict["POS"],
                        "texts_neg": texts_dict["NEG"]
                    })
                # else:
                #     print(path)
        return corpus
                

if __name__ == "__main__":
    import pprint
    data = DataLoader(["hake"])    
    pprint.pprint (data.data)
