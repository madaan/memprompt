import logging
import random
import datetime
import pandas as pd
import json
import time

from src.openai_wrapper import OpenaiWrapperWithMemory, Prompt
from src.task_handler import TaskHandler


class InteractionHandler(object):
    def __init__(self, prompt_path: str, getting_clarification_probability: float, job_id: str):
        self.prompt = Prompt(prompt_path)
        self.lm_hander = OpenaiWrapperWithMemory(prompt=self.prompt, memory=None)

        self.task_handler = TaskHandler(job_id)
        self.getting_clarification_probability = getting_clarification_probability
        self.job_id = job_id

    def run(self, task_file: str):
        # load the task file
        tasks = self.load_tasks(task_file)

        conversation_history = []
        num_correct = 0

        for task_idx, task in enumerate(tasks):
            try:
                # get a task
                question, expected_answer = self.task_handler.get_qa_from_task(task)
                start_time = time.monotonic_ns()
                logging.info(
                    f"[INTERACTION] {task_idx} > Question: {question} | Expected answer: {expected_answer}"
                )
                # get a question
                response = self.lm_hander.send_query_with_custom_prompt(
                    question=question, prompt=self.prompt
                )
                # get the answer
                generated_answer = self.lm_hander.parse_response(response)
                is_correct = self.task_handler.check_answer(answer=generated_answer, task=task)
                logging.info(
                    f"[INTERACTION] {task_idx} > Generated Answer: {generated_answer} | Correct: {is_correct}"
                )
                if not is_correct:
                    prob = random.random()
                    if prob < self.getting_clarification_probability:
                        clarification = self.task_handler.get_clarification(task)
                        self.prompt.add_example_to_prompt(
                            question=f"{question} | {clarification}", answer=expected_answer
                        )
                        logging.info(f"Prompt updated to {self.prompt.text}")
                else:
                    num_correct += 1
                correct_so_far = round(num_correct * 100 / (task_idx + 1), 2)
                logging.info(f"[INTERACTION] Correct so far: {correct_so_far}%")
                conversation_history.append(
                    {
                        "question": question,
                        "expected_answer": expected_answer,
                        "generated_answer": generated_answer,
                        "response": response,
                        "is_correct": is_correct,
                        "correct_so_far": correct_so_far,
                        "idx": task_idx,
                        "prompt_length": len(self.prompt.text.split()),
                        "prompt": self.prompt.text,
                        "elapsed_time": time.monotonic_ns() - start_time,
                    }
                )
                logging.info(f"[INTERACTION] Prompt len: {len(self.prompt.text.split())}")
            except Exception as e:
                logging.error(f"[INTERACTION] Error: {e}")
                conversation_history.append(
                    {
                        "question": question,
                        "expected_answer": expected_answer,
                        "generated_answer": generated_answer,
                        "exception": True,
                        "is_correct": False,
                        "correct_so_far": correct_so_far,
                        "idx": task_idx,
                        "prompt_length": len(self.prompt.text.split()),
                        "prompt": self.prompt.text,
                        "elapsed_time": time.monotonic_ns() - start_time,
                    }
                )
                continue
            logging.info("-" * 80)
            logging.info("-" * 80)

        # save the conversation history

        df = pd.DataFrame(conversation_history)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        df.to_json(
            f"logs/growing_prompt_task_type={self.job_id}_num_tasks={len(tasks)}_clarification_prob={self.getting_clarification_probability}_ts={timestamp}.jsonl",
            orient="records",
            lines=True,
        )

    def load_tasks(self, task_file: str):
        tasks = []
        with open(task_file, "r") as f:
            for line in f:
                tasks.append(json.loads(line))
        return tasks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import pathlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, default="data/prompts/prompt.txt")
    parser.add_argument("--task_file", type=str)
    parser.add_argument("--getting_clarification_probability", type=float, default=0.5)
    parser.add_argument("--job_id", type=str, default="")
    args = parser.parse_args()
    handler = InteractionHandler(
        args.prompt_path,
        job_id=args.job_id,
        getting_clarification_probability=args.getting_clarification_probability,
    )
    pathlib.Path(f"logs/").mkdir(parents=True, exist_ok=True)
    handler.run(args.task_file)
    print("Done")
