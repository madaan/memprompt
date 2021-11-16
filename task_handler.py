import logging
import json
import pandas as pd
import random
from collections import Counter

from memprompt.templates import templates


logging.basicConfig(level=logging.INFO)


class TaskHandler(object):
    ### randomly returns a task
    def __init__(
        self,
        template_class,
        load_raw_data: bool = False,
        task_types = ["syn", "ant", "hom", "defn", "sent"],
        wordnet_raw_data_path: str = "data/wordnet/raw.jsonl",
        commongen_raw_data_path: str = "data/commongen/raw.jsonl",
        defn_raw_data_path: str = "data/defn/raw.jsonl",
    ):
        self.task_types = task_types
        if load_raw_data:
            self.wordnet_tasks = pd.read_json(wordnet_raw_data_path, orient="records", lines=True)
            self.commongen_tasks = pd.read_json(
                commongen_raw_data_path, orient="records", lines=True
            )
            self.defn_tasks = pd.read_json(defn_raw_data_path, orient="records", lines=True)
            # concatenate all of them
            self.tasks = pd.concat([self.wordnet_tasks, self.commongen_tasks, self.defn_tasks])
            self.tasks = self.tasks.sample(frac=1)

            # make a hashmap from task type to task data
            self.task_map = {}
            for task_type in self.task_types:
                self.task_map[task_type] = self.tasks[self.tasks["type"] == task_type]

        self.templates = templates[template_class]

        logging.info("Loaded {} tasks".format(len(self.task_types)))

        # make a hashmap from task type to list of template ids
        self.task_type_to_template_ids = {task_type: [] for task_type in self.task_types}
        for template in self.templates:
            if template["type"] in self.task_type_to_template_ids:
                self.task_type_to_template_ids[template["type"]].append(template["template_id"])

        # make a hashmap from template id to template
        self.template_id_to_template_map = {}
        for template in self.templates:
            self.template_id_to_template_map[template["template_id"]] = template

    def get_random_task(self):
        """
        {
          "meta": {
              "word": "squinancy",
              "definition": "A European perennial herb (Asperula cynanchica) with narrowly linear whorled leaves"
          },
          "type": "defn"
          }
        """
        # randomly select a task type
        task_type = random.choice(self.task_types)
        # randomly select a task from that type
        task = self.task_map[task_type].sample(n=1).to_dict(orient="records")[0]
        return task

    def get_qa_from_task(self, task):
        template = self.template_id_to_template_map[task["template_id"]]
        question, answer = self.prepare_task(task, template)
        return question, answer

    def create_tasks(self, n):
        tasks = []
        for i in range(n):
            task = self.get_random_task()
            template_id = random.choice(self.task_type_to_template_ids[task["type"]])
            task["template_id"] = template_id
            tasks.append(task)
        return tasks

    def get_clarification(self, task):
        template = self.template_id_to_template_map[task["template_id"]]
        return template["clarification"]

    def get_template(self, task_type):
        template = random.choice(self.template_map[task_type])
        return template

    def prepare_syn(self, task, template):
        word1, word2 = task["meta"]["word1"], task["meta"]["word2"]
        question = template["question"](word1)
        answer = template["answer"](word1, word2)
        return question, answer

    def prepare_ant(self, task, template):
        word1, word2 = task["meta"]["word1"], task["meta"]["word2"]
        question = template["question"](word1)
        answer = template["answer"](word1, word2)
        return question, answer

    def prepare_defn(self, task, template):
        word = task["meta"]["word"]
        definition = task["meta"]["definition"]

        question = template["question"](word)
        answer = template["answer"](word, definition)
        return question, answer

    def prepare_sent(self, task, template):
        word = task["meta"]["word"]
        sentence = task["meta"]["sentence"]
        question = template["question"](word)
        answer = template["answer"](word, sentence)
        return question, answer

    def prepare_hom(self, task, template):
        word1, word2 = task["meta"]["word1"], task["meta"]["word2"]
        question = template["question"](word1)
        answer = template["answer"](word1, word2)
        return question, answer

    def prepare_task(self, task, template):
        if task["type"] == "syn":
            return self.prepare_syn(task, template)
        elif task["type"] == "ant":
            return self.prepare_ant(task, template)
        elif task["type"] == "defn":
            return self.prepare_defn(task, template)
        elif task["type"] == "sent":
            return self.prepare_sent(task, template)
        elif task["type"] == "hom":
            return self.prepare_hom(task, template)
        else:
            raise ValueError("Unknown task type")

    def check_syn_answer(self, answer):
        return "synonym" in answer

    def check_ant_answer(self, answer):
        return "antonym" in answer

    def check_defn_answer(self, answer):
        return "definition" in answer

    def check_sent_answer(self, answer):
        return "sentence" in answer and "[" in answer

    def check_hom_answer(self, answer):
        return "homonym" in answer

    def check_answer(self, answer, task):
        if task["type"] == "syn":
            return self.check_syn_answer(answer)
        elif task["type"] == "ant":
            return self.check_ant_answer(answer)
        elif task["type"] == "defn":
            return self.check_defn_answer(answer)
        elif task["type"] == "sent":
            return self.check_sent_answer(answer)
        elif task["type"] == "hom":
            return self.check_hom_answer(answer)
        else:
            raise ValueError("Unknown task type")


def dump_tasks(n, template_class, task_types):
    if template_class not in ["linguistic", "hin", "pun"]:
        raise ValueError("Unknown template class, must be linguistic, hin, or pun")

    task_stream = TaskHandler(template_class, task_types=task_types, load_raw_data=True)

    tasks = task_stream.create_tasks(n)
    # distribution of task types
    logging.info(Counter(task["type"] for task in tasks))
    # distribution of template ids
    logging.info(Counter(task["template_id"] for task in tasks))
    # write to a file
    pathlib.Path(f"tasks/{template_class}").mkdir(parents=True, exist_ok=True)
    # is the input different from the standard task types?
    tasks_meta = "tasks" if set(task_types).difference(set(DEFAULT_TASK_TYPES)) == 0  else f"tasks_{','.join(task_types)}"
    out_fp = f"tasks/{template_class}/{tasks_meta}_{n}.jsonl"
    with open(out_fp, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")


def make_samples(
    n: int,
    template_class: str,
    outpath: str,
    wordnet_raw_data_path: str,
    commongen_raw_data_path: str,
    defn_raw_data_path: str,
):
    task_handler = TaskHandler(
        template_class=template_class,
        load_raw_data=True,
        wordnet_raw_data_path=wordnet_raw_data_path,
        commongen_raw_data_path=commongen_raw_data_path,
        defn_raw_data_path=defn_raw_data_path,
    )
    res = []
    tasks = task_handler.create_tasks(n)
    for task in tasks:
        ques, ans = task_handler.get_qa_from_task(task)
        res.append({"ques": ques, "ans": ans, "type": task["type"]})

    pd.DataFrame(res).to_json(outpath, orient="records", lines=True)


if __name__ == "__main__":
    import sys
    import pathlib
    import argparse
    DEFAULT_TASK_TYPES = ["syn", "ant" , "hom", "defn", "sent"]
    args = argparse.ArgumentParser()
    args.add_argument("--template_class", type=str, default="linguistic")
    args.add_argument("--n", type=int, default=100)
    args.add_argument("--task_types", type=str, default=",".join(DEFAULT_TASK_TYPES))
    args.add_argument("--outpath", type=str, default="data/tasks.jsonl")
    args.add_argument("--wordnet_raw_data_path", type=str, default="data/wordnet_raw_data.jsonl")
    args.add_argument("--commongen_raw_data_path", type=str, default="data/commongen_raw_data.jsonl")
    args.add_argument("--defn_raw_data_path", type=str, default="data/defn_raw_data.jsonl")
    args.add_argument("--dump", action="store_true")
    args.add_argument("--make_samples", action="store_true")
    args = args.parse_args()

    if args.dump:
        dump_tasks(args.n, args.template_class, task_types=args.task_types.split(","))
    elif args.make_samples:
        make_samples(
            n=args.n,
            template_class=args.template_class,
            outpath=args.outpath,
            wordnet_raw_data_path=args.wordnet_raw_data_path,
            commongen_raw_data_path=args.commongen_raw_data_path,
            defn_raw_data_path=args.defn_raw_data_path,
        )
