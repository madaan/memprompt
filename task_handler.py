import logging
import json
import pandas as pd
import random
from collections import Counter
from typing import List


from memprompt.templates import templates
from memprompt.templates_gpt3_wordtasks import templates_gpt3_wordtasks

templates.update(templates_gpt3_wordtasks)

logging.basicConfig(level=logging.INFO)


class TaskHandler(object):
    ### randomly returns a task
    def __init__(
        self, template_class, task_files: List = None, load_raw_data: bool = False,
    ):
        if load_raw_data:
            raw_datasets = []
            for task_file in task_files:
                raw_datasets.append(pd.read_json(task_file, lines=True, orient="records"))
            self.tasks = pd.concat(raw_datasets)
            self.tasks = self.tasks.sample(frac=1)
            task_types = list(self.tasks["type"].unique())
            logging.info(f"Loaded {len(self.tasks)} tasks {task_types}")
            # make a hashmap from task type to task data
            self.task_map = {}
            for task_type in task_types:
                self.task_map[task_type] = self.tasks[self.tasks["type"] == task_type]
        self.word_level_tasks = set(
            ["syn", "ant", "hom", "cyc", "anag1", "anag2", "rev", "randsym"]
        )

        self.templates = templates[template_class]
        self.task_types = list(set([template["type"] for template in self.templates]))

        # make a hashmap from task type to list of template ids
        self.task_type_to_template_ids = {task_type: [] for task_type in self.task_types}
        for template in self.templates:
            """
            Each template in the template list has the following form:
                {
                "type": "cyc",
                "template_id": "cyc0",
                "question": lambda word1: f"Find the right word given this cycled word: < {word1} > ?",
                "clarification": "clarification: when I want you to fix cycled word, I mean cycled.",
                "answer": lambda word1, word2: f"the uncycled version of {word1} is {word2}",
                },
            """
            task_type = template["type"]
            template_id = template["template_id"]
            self.task_type_to_template_ids[task_type].append(template_id)

        logging.info(self.task_type_to_template_ids)

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

    def prepare_task(self, task, template):
        if template["type"] in self.word_level_tasks:
            return self.prepare_word_task(task, template)
        elif task["type"] == "defn":
            return self.prepare_defn(task, template)
        elif task["type"] == "sent":
            return self.prepare_sent(task, template)
        else:
            raise ValueError("Unknown task type")

    def prepare_syn(self, task, template):
        return self.prepare_word_task(task, template)

    def prepare_ant(self, task, template):
        return self.prepare_word_task(task, template)

    def prepare_hom(self, task, template):
        return self.prepare_word_task(task, template)

    def prepare_word_task(self, task, template):
        word1, word2 = task["meta"]["word1"], task["meta"]["word2"]
        question = template["question"](word1)  # generate question using the template lambda
        answer = template["answer"](word1, word2)  # generate answer using the template lambda
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

    def check_answer(self, answer, task):
        check_name = f"check_{task['type']}_answer"
        try:
            return getattr(self, check_name)(answer)
        except:
            raise ValueError(f"Unknown task type: {task['type']}")

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

    def check_cyc_answer(self, answer):
        return "uncycled version" in answer

    def check_anag1_answer(self, answer):
        return "anagram 1" in answer

    def check_anag2_answer(self, answer):
        return "anagram 2" in answer

    def check_rev_answer(self, answer):
        return "after reversing" in answer

    def check_randsym_answer(self, answer):
        return "word after removing symbols" in answer


def dump_tasks(n, template_class, task_files):
    if template_class not in ["linguistic", "hin", "pun", "synthetic_gpt3"]:
        raise ValueError(f"Unknown template class, must be in {template_class}")

    task_stream = TaskHandler(template_class, load_raw_data=True, task_files=task_files)

    tasks = task_stream.create_tasks(n)
    # distribution of task types
    logging.info(Counter(task["type"] for task in tasks))
    # distribution of template ids
    logging.info(Counter(task["template_id"] for task in tasks))
    # write to a file
    pathlib.Path(f"tasks/{template_class}").mkdir(parents=True, exist_ok=True)
    # is the input different from the standard task types?

    with open(f"tasks/{template_class}/tasks_{n}.jsonl", "w") as f:
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

    args = argparse.ArgumentParser()
    args.add_argument("--template_class", type=str)
    args.add_argument("--n", type=int, default=100)
    args.add_argument("--task_types", type=str)
    args.add_argument("--outpath", type=str, default="data/tasks.jsonl")
    args.add_argument("--raw_files", type=str, help="comma separated list of task files")
    args.add_argument("--dump", action="store_true")
    args.add_argument("--make_samples", action="store_true")
    args = args.parse_args()

    raw_files = [path.strip() for path in args.raw_files.split(",")]

    if args.dump:
        dump_tasks(args.n, args.template_class, raw_files)
    elif args.make_samples:
        make_samples(
            n=args.n,
            template_class=args.template_class,
            outpath=args.outpath,
            wordnet_raw_data_path=args.wordnet_raw_data_path,
            commongen_raw_data_path=args.commongen_raw_data_path,
            defn_raw_data_path=args.defn_raw_data_path,
        )
