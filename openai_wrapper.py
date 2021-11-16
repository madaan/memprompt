import logging
import os
import random
import openai
import re

from memprompt.closest_match_memory import ClosestMatchMemory
from memprompt.semantic_memory import SemanticMatchMemory

openai.api_key = os.getenv("OPENAI_API_KEY")


class Prompt(object):
    def __init__(self, prompt_path: str, max_examples: int = 20, max_length: int = 1300):
        self.prompt_path = prompt_path
        self.SEP = "\n\n###\n\n/"
        self.END = "END"
        self.max_examples = max_examples
        self.max_length = max_length
        with open(prompt_path, "r") as f:
            self.text = f.read()
            self.text = self.text.replace("[SEP]", self.SEP)

        self.base_text = self.text

        self.examples_added_so_far = []

    def add_example_to_prompt(self, question: str, answer: str):
        self.examples_added_so_far.append((question, answer))
        if (len(self.examples_added_so_far) > self.max_examples) or (
            len(self.text.split()) > self.max_length
        ):
            self.shrink_prompt()
        else:
            qa = f"{question}{self.SEP}{answer} {self.END}"
            self.text = f"{self.text}{self.SEP}{qa}"

    def shrink_prompt(self):
        self.text = self.base_text
        for question, answer in self.examples_added_so_far[-self.max_examples :]:
            qa = f"{question}{self.SEP}{answer} {self.END}"
            self.text = f"{self.text}{self.SEP}{qa}"


class OpenaiWrapperWithMemory(object):
    def __init__(self, memory, prompt_path: str = None, prompt: Prompt = None):
        assert prompt_path or prompt, "Either prompt_path or prompt should be provided"
        if prompt_path:
            self.prompt = Prompt(prompt_path)
        else:
            self.prompt = prompt  # the query should provide the prompt
        self.memory = memory

    @staticmethod
    def create_with_closest_match_memory(prompt_path: str = None, prompt: str = None):
        memory = ClosestMatchMemory(OpenaiWrapperWithMemory._process_key_for_closest)
        return OpenaiWrapperWithMemory(memory=memory, prompt_path=prompt_path, prompt=prompt)

    @staticmethod
    def _process_key_for_closest(key):
        key = re.sub("<.*>", "", key).lower().strip()
        # remove double spaces
        key = re.sub(" +", " ", key)
        # remove the punctuation
        key = re.sub("[^a-zA-Z0-9 ]", "", key)
        # # remove the stop words
        # key = " ".join([w for w in key.split() if w not in sw])
        return key.strip()

    @staticmethod
    def create_with_semantic_memory(
        prompt_path: str = None,
        prompt: str = None,
        checkpoint_path: str = None,
        model_name: str = "bert-base-uncased",
    ):
        memory = SemanticMatchMemory(
            key_process_func=lambda key: key.lower().strip(),
            checkpoint_path=checkpoint_path,
            model_name=model_name,
        )
        return OpenaiWrapperWithMemory(memory=memory, prompt_path=prompt_path, prompt=prompt)



    def send_memory_assisted_query(self, question: str):
        memory_metadata = {}

        #  step 1 : check if a feedback is available for this question
        closest_key, maybe_enriched_question, closest_key_score = self.enrich_with_memory(question)

        memory_metadata["memory_used"] = False
        memory_metadata["memory_size"] = self.get_memory_size()
        memory_metadata["content"] = self.get_memory()
        if maybe_enriched_question != question:
            memory_metadata["memory_used"] = True
            memory_metadata["match_score"] = closest_key_score
            memory_metadata["query_for_memory_lookup"] = self.memory.key_process_func(question)
            memory_metadata["memory_key"] = closest_key
            memory_metadata["memory_value"] = maybe_enriched_question.split("|")[-1].strip()

            logging.info(f"[LM] Question changed to {maybe_enriched_question}")
        #  step 2 : add prompt to question
        question_with_prompt = (
            f"{self.prompt.text}{self.prompt.SEP}{maybe_enriched_question}{self.prompt.SEP}"
        )
        #  step 3 : send question to openai
        response = self.make_openai_api_call(question_with_prompt)
        return response, memory_metadata

    def send_query_with_custom_prompt(self, question: str, prompt: Prompt):
        #  step 1 : add prompt to question
        question_with_prompt = f"{prompt.text}{prompt.SEP}{question}{prompt.SEP}"
        #  step 3 : send question to openai
        response = self.make_openai_api_call(question_with_prompt)
        return response

    def enrich_with_memory(self, question):
        closest_key, maybe_clarification, score = self.memory.get_closest(question)
        if maybe_clarification:  # if there is a clarification available, use it
            logging.info(f"[LM] Found a clarification for {question} : {maybe_clarification} with score = {score}")
            return closest_key, f"{question} | {maybe_clarification}", score
        else:
            return None, question, None

    def add_feedback(self, question, clarification):
        self.memory[question] = clarification
        logging.info(f"[LM] Thanks, feedback noted {question} : {clarification}")

    def make_openai_api_call(self, prompt: str):
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.7,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=[self.prompt.END],
        )
        return response

    def parse_response(self, response):
        text = response["choices"][0]["text"]
        if self.prompt.SEP in text:
            text = text.split(self.prompt.SEP)[-1]
        return text

    def get_memory(self):
        return str(self.memory)

    def get_memory_size(self):
        return len(self.memory)
