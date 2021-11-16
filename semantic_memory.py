import logging
from typing import List
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation, util
from collections import OrderedDict
from torch import nn
import torch
from torch.utils.data import DataLoader
import pandas as pd
import random
from sklearn.model_selection import train_test_split

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

class SemanticMatchMemory(object):
    def __init__(self, model_name: str, checkpoint_path: str = None, match_threshold: float = 0.9,
    key_process_func: callable = None, mode: str = "eval"):
        if mode == "eval":
            assert checkpoint_path is not None, "checkpoint_path must be provided for eval mode"
        word_embedding_model = models.Transformer(model_name, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=256,
            activation_function=nn.Tanh(),
        )
        logging.info("Initializing model...")
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device="cpu")
        logging.info("Model initialized")
        self.memory = OrderedDict()
        self.memory_embedding = []
        self.match_threshold = match_threshold
        self.key_process_func = key_process_func
        if checkpoint_path:
            logging.info(f"Loading model from checkpoint {checkpoint_path}")
            self.load_model(checkpoint_path)

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def parse_key(self, key: str) -> str:
        if self.key_process_func:
            return self.key_process_func(key)
        else:
            return key

    def train(self, train_path: str, test_path: str, outpath: str, batch_size: int = 64, epochs: int = 1):

        train_examples = self.read_examples(train_path)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)

        eval_sentences1, eval_sentences2, eval_scores = self.read_examples(
            test_path, read_for_eval=True
        )
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            eval_sentences1, eval_sentences2, eval_scores
        )

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=500,
            evaluator=evaluator,
            evaluation_steps=500,
        )

        # save model
        torch.save(self.model.state_dict(), outpath)

    def read_examples(self, path: str, read_for_eval: bool = False) -> List[InputExample]:
        examples_df = pd.read_json(path, orient="records", lines=True)
        examples = []
        sentences1, sentences2 = [], []
        scores = []
        for i, row in examples_df.iterrows():
            if read_for_eval:
                sentences1.append(row["sentence1"])
                sentences2.append(row["sentence2"])
                scores.append(float(row["label"]))
            else:
                examples.append(
                    InputExample(texts=[row["sentence1"], row["sentence2"]], label=float(row["label"]))
                )
        if read_for_eval:
            return sentences1, sentences2, scores
        return examples

    def __setitem__(self, key, value):
        key = self.parse_key(key)
        self.memory[key] = value
        self.memory_embedding.append(self.model.encode(key))

    
    def get_closest(self, key):
        if len(self) == 0:
            return None, None, None
        key = self.parse_key(key)
        closest_key, closest_key_score = self._find_closest_key(key)
        if closest_key:
            return closest_key, self[closest_key], closest_key_score
        else:
            return None, None, None
    
    @torch.no_grad()
    def _find_closest_key(self, query):
        query_embedding = self.model.encode(query)
        
        # find closest key
        scores = util.cos_sim(query_embedding, self.memory_embedding).squeeze(0)
        max_score_idx = torch.argmax(scores).item()
        closest_key = list(self.memory.keys())[max_score_idx]

        # get closest key score
        closest_key_score = scores[max_score_idx].item()

        if closest_key_score > self.match_threshold:
            return closest_key, closest_key_score
        else:
            return None, None


    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]

    def __str__(self) -> str:
        return " || ".join([f"{k}: {v}" for (k, v) in self.memory.items()])
        

def make_binary_classification_data(path: str, outpath: str, n_samples=1000):
    data = pd.read_json(path, orient="records", lines=True)
    # create five dataframes, one for each type
    possible_types = set(data["type"].tolist())
    possible_types_list = list(possible_types)
    dataframes = {}
    for type in possible_types:
        dataframes[type] = data[data["type"] == type]
    # first create examples of similar sentences, by taking two random sentences from each type
    examples = []
    for type in possible_types:
        for _ in range(n_samples):
            samples = dataframes[type].sample(2)
            examples.append(
                {
                    "sentence1": samples.iloc[0]["ques"],
                    "sentence2": samples.iloc[1]["ques"],
                    "label": 1.,
                }
            )
    
    all_type_pairs = []
    for type1 in possible_types:
        for type2 in possible_types:
            if type1 != type2:
                all_type_pairs.append((type1, type2))
    # then create examples of dissimilar sentences, by taking two random sentences from each type
    for _ in range(n_samples * len(possible_types)):
        type1, type2 = random.choice(all_type_pairs)
        sample1 = dataframes[type1].sample(1)
        sample2 = dataframes[type2].sample(1)

        examples.append(
            {"sentence1": sample1.iloc[0]["ques"], "sentence2": sample2.iloc[0]["ques"], "label": 0.}
        )

    random.shuffle(examples)
    all_data = pd.DataFrame(examples)
    # delete the duplicate rows
    print(f"Deleting duplicate rows (before: {len(all_data)})")
    all_data = all_data.drop_duplicates(subset=["sentence1", "sentence2"])
    print(f"Deleting duplicate rows (after: {len(all_data)})")
    train, test = train_test_split(all_data, test_size=0.1)
    train.to_json(f"{outpath}/train.jsonl", orient="records", lines=True)
    test.to_json(f"{outpath}/test.jsonl", orient="records", lines=True)


def unit_test_trained_memory(args):
    trained_memory = SemanticMatchMemory(model_name=args.model_name, checkpoint_path=args.checkpoint_path)

    trained_memory["< cup > can be used how"] = "c1"
    trained_memory["how can <harbour> be used"] = "c2"
    trained_memory["how can harbour be used"] = "c3"
    trained_memory["how can < harbour > be used"] = "c4"
    trained_memory["< port > can be used how"] = "c5"
    trained_memory["< harbor > can be used how"] = "c6"


    trained_memory["what rings like < papaya > ?"] = "clarification | rings like means homonym"
    trained_memory["use < carrot > in a sentence"] = "clarification | use means give a sentence with the word"
    trained_memory["what is like < apple > ?"] = "clarification | like means synonym"
    # pretty print the memory
    for key, value in trained_memory.memory.items():
        print(f"{key}: {value}")

    # now check if the closest match is found
    closest_key, matched_value, matched_score = trained_memory.get_closest("what rings like < tomato > ?")
    assert closest_key == "what rings like < papaya > ?"
    assert matched_value == "clarification | rings like means homonym", f"{trained_memory.get_closest('what rings like < tomato > ?')}"
    
    closest_key, matched_value, matched_score = trained_memory.get_closest("use < tomato > in a sentence")
    assert closest_key == "use < carrot > in a sentence"
    assert matched_value == "clarification | use means give a sentence with the word"

    # closest_key, matched_value = trained_memory.get_closest("what sounds like < papaya > ?")
    # assert closest_key == "what rings like < papaya > ?"
    # assert matched_value == "clarification | rings like means homonym"

    closest_key, matched_value, matched_score = trained_memory.get_closest("what is akin to < fleetwood mac > ?")
    assert closest_key == "what is like < apple > ?"
    assert matched_value == "clarification | like means synonym"

    q = "Please try to use < harbour > in some way"
    closest_key, matched_value, matched_score = trained_memory.get_closest(q)
    print(f"query= {q} , matched key = {closest_key}")
    assert matched_value in ["c2", "c3", "c4", "c5", "c6"]

    print("All tests passed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--train_path", type=str, default="data/sim/train.jsonl")
    parser.add_argument("--test_path", type=str, default="data/sim/test.jsonl")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_path", type=str)
    parser.add_argument(
        "--raw_data_path",
        type=str,
        help="path to raw data, used to create binary data for similarity classification",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="number of samples to create for binary classification",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="path to checkpoint to load",
    )

    parser.add_argument(
        "--make_binary",
        action="store_true",
        help="create binary data for similarity classification",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="train the model",
    )

    parser.add_argument(
        "--unit_test",
        action="store_true",
        help="run unit tests",
    )
    

    args = parser.parse_args()
    if args.make_binary:
        make_binary_classification_data(
            path=args.raw_data_path, outpath=args.out_path, n_samples=args.n_samples
        )
    elif args.train:
        sim_model = SemanticMatchMemory(model_name=args.model_name)
        sim_model.train(
            train_path=args.train_path,
            test_path=args.test_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            outpath=args.out_path,
        )
    elif args.unit_test:
        unit_test_trained_memory(args)
    else:
        raise NotImplementedError("not implemented yet")
