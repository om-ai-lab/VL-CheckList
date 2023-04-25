import json
import os
import random
import time

import yaml
from tqdm import tqdm

from vl_checklist.data_loader import DataLoader
from vl_checklist.utils import add_caption, chunks


class Evaluate(object):
    def __init__(self, config_file, model) -> None:
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.cur_dir = os.path.realpath(os.curdir)
        m = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
        self.batch_size = m["BATCH_SIZE"]
        self.model = model
        self.max_num = m["MAX_NUM"]
        self.data_names = m["DATA"]["TEST_DATA"]
        self.task = m["TASK"]
        self.types = m["DATA"]["TYPES"]
        self.dir = m["OUTPUT"]["DIR"]
        self.sample_num = m["OUTPUT"]["NUM"]
        self.model_name = model.model_name()

    def start(self):
        print(f"Configs: model = {self.model_name} | task = {self.task}  | datasets = {self.data_names}")
        for data_type in self.types:
            self.eval(data_type=data_type)

    def eval(self, data_type):
        max_number = self.max_num
        d = DataLoader(self.data_names, data_type, self.task)
        results = {}
        index = 0
        if self.task == "itm":
            for name in d.data:
                sample_true = []
                sample_false = []
                # true false positive negative
                tp, tn, fp, fn = 0, 0, 0, 0
                if max_number:
                    d.data[name] = d.data[name][:max_number]
                starttime = time.time()
                for batch in tqdm(
                    chunks(d.data[name], self.batch_size),
                    desc="Progress",
                    ncols=100,
                    total=int(len(d.data[name]) / self.batch_size),
                ):
                    images = [z["path"] for z in batch]
                    texts = [z["texts"][0] for z in batch]
                    # texts = [random.choice(z["texts"]) for z in batch]
                    labels = [z["label"] for z in batch]
                    try:
                        result = self.model.predict(images, texts, src_type="local")
                    except Exception as e:
                        print(e)
                        continue
                    predictions = [
                        1 if z[1] > 0.5 else 0 for i, z in enumerate(result[0])
                    ]
                    for p, l, i, t, r in zip(
                        predictions, labels, images, texts, result[0]
                    ):
                        if p == 0 and l == 0:
                            tn += 1
                            sample_true.append(
                                {
                                    "img_path": i,
                                    "score": round(r[1], 4),
                                    "text": t,
                                    "result": "correct",
                                }
                            )
                        elif p == 1 and l == 1:
                            tp += 1
                            sample_true.append(
                                {
                                    "img_path": i,
                                    "score": round(r[1], 4),
                                    "text": t,
                                    "result": "correct",
                                }
                            )
                        elif p == 1 and l == 0:
                            fp += 1
                            sample_false.append(
                                {
                                    "img_path": i,
                                    "score": round(r[1], 4),
                                    "text": t,
                                    "result": "incorrect",
                                }
                            )
                        else:
                            fn += 1
                            sample_false.append(
                                {
                                    "img_path": i,
                                    "score": round(r[1], 4),
                                    "text": t,
                                    "result": "incorrect",
                                }
                            )
                endtime = time.time()
                print(tp, tn, fp, fn)
                precision = float(tp) / (tp + fp)
                recall = float(tp) / (tp + fn)
                accuracy = float(tp + tn) / (tp + tn + fp + fn)
                f1 = 2 * (precision * recall) / (precision + recall)
                results[
                    name
                ] = f"precision: {round(precision, 4)}, recall: {round(recall, 4)}, f1: {round(f1, 4)}, acc: {round(accuracy, 4)}"

                file_name = data_type.replace("/", "_")
                sample_t = random.sample(sample_true, self.sample_num)
                sample_f = random.sample(sample_false, self.sample_num)

                sample_path = os.path.join(
                    self.cur_dir, self.dir, "itm", "sample", f"{file_name}_{name}"
                )
                if not os.path.exists(sample_path):
                    os.makedirs(sample_path)
                with open(
                    os.path.join(
                        self.cur_dir, self.dir, "itm", f"{file_name}_{name}.json"
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(
                        {
                            "sample_correct_outputs": sample_t,
                            "sample_incorrect_outputs": sample_f,
                            "total_acc": round(accuracy, 4),
                            "number_of_data": len(d.data[name]),
                            "model_name": self.model_name,
                            "task": self.task,
                            "eval_time": endtime - starttime,
                        },
                        f,
                    )

                for n, i in enumerate(zip(sample_t, sample_f)):
                    add_caption(
                        i[0]["img_path"],
                        "text:" + i[0]["text"],
                        "score:" + str(i[0]["score"]),
                        None,
                        None,
                        sample_path,
                        f"cor-{n+1}",
                    )
                    add_caption(
                        i[1]["img_path"],
                        "text:" + i[1]["text"],
                        "score:" + str(i[1]["score"]),
                        None,
                        None,
                        sample_path,
                        f"incor-{n+1}",
                    )

        elif self.task == "itc":
            for name in d.data:
                print(f"Evaluating {name}/{data_type} data")
                sample_true = []
                sample_false = []
                num_t, num_f = 0, 0
                if max_number:
                    d.data[name] = d.data[name][: int(max_number / 2)]
                starttime = time.time()
                for batch in tqdm(
                    chunks(d.data[name], self.batch_size),
                    desc="Progress",
                    ncols=100,
                    total=int(len(d.data[name]) / self.batch_size),
                ):
                    images = [z["path"] for z in batch]
                    texts_pos = [z["texts_pos"][0] for z in batch]
                    texts_neg = [z["texts_neg"][0] for z in batch]
                    # print(
                    #     f"pos sz: {len(texts_pos)}, neg sz: {len(texts_neg)}, img sz: {len(images)}"
                    # )
                    # try:
                    result_pos = self.model.predict(images, texts_pos, src_type="local")
                    result_neg = self.model.predict(images, texts_neg, src_type="local")
                    # print(
                    #     f"result pos: {result_pos['probs']}, result neg: {result_neg['probs']}"
                    # )
                    # except Exception as e:
                    #    print(e)
                    #    continue

                    result_t1 = zip(result_pos["probs"], result_neg["probs"])
                    result_tmp = list(result_t1)
                    for i in range(len(result_tmp)):
                        index = index + 1
                        if result_tmp[i][0] > result_tmp[i][1]:
                            sample_true.append(
                                {
                                    "img_path": images[i],
                                    "pos_score": round(result_tmp[i][0], 4),
                                    "pos_txt": texts_pos[i],
                                    "neg_score": round(result_tmp[i][1], 4),
                                    "neg_txt": texts_neg[i],
                                    "result": "correct",
                                }
                            )
                            num_t += 1
                        else:
                            sample_false.append(
                                {
                                    "img_path": images[i],
                                    "pos_score": round(result_tmp[i][0], 4),
                                    "pos_txt": texts_pos[i],
                                    "neg_score": round(result_tmp[i][1], 4),
                                    "neg_txt": texts_neg[i],
                                    "result": "incorrect",
                                }
                            )
                            num_f += 1
                endtime = time.time()
                accuracy = float(num_t) / (num_t + num_f)
                results[name] = f"acc: {round(accuracy, 4)}"
                file_name = data_type.replace("/", "_")
                sample_t = random.sample(sample_true, self.sample_num)
                sample_f = random.sample(sample_false, self.sample_num)

                sample_path = os.path.join(
                    self.cur_dir, self.dir, "itc", "sample", f"{file_name}_{name}"
                )
                if not os.path.exists(sample_path):
                    os.makedirs(sample_path)
                with open(
                    os.path.join(
                        self.cur_dir, self.dir, "itc", f"{file_name}_{name}.json"
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    result_meta = {
                        "sample_correct_outputs": sample_t,
                        "sample_incorrect_outputs": sample_f,
                        "total_acc": round(accuracy, 4),
                        "number_of_data": len(d.data[name]),
                        "model_name": self.model_name,
                        "task": self.task,
                        "eval_time": endtime - starttime,
                    }
                    json.dump(result_meta, f, indent=4)

                for n, i in enumerate(zip(sample_t, sample_f)):
                    add_caption(
                        i[0]["img_path"],
                        "pos_text:" + i[0]["pos_txt"],
                        "pos_score:" + str(i[0]["pos_score"]),
                        "neg_text:" + i[0]["neg_txt"],
                        "neg_score:" + str(i[0]["neg_score"]),
                        sample_path,
                        f"cor-{n+1}",
                    )
                    add_caption(
                        i[1]["img_path"],
                        "pos_text:" + i[1]["pos_txt"],
                        "pos_score:" + str(i[1]["pos_score"]),
                        "neg_text:" + i[1]["neg_txt"],
                        "neg_score:" + str(i[1]["neg_score"]),
                        sample_path,
                        f"incor-{n+1}",
                    )

        return results
