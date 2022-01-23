import json
from functools import reduce
import logging
import os
import re
import tempfile
from os import listdir
from os.path import isfile, join
from glob import glob
from typing import List, Dict

import pandas as pd


logging.basicConfig(level=logging.INFO)

def get_all_stopwords():
    from nltk.corpus import stopwords
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    all_stopwords = nlp.Defaults.stop_words
    sw = set(stopwords.words("english")).union(set(all_stopwords))
    return sw

def parse_one_conversation_log(jsonl_path) -> List[str]:
    return [log_entry["correct_so_far"] for log_entry in read_jsonl(jsonl_path)]

def matches(pattern, s):
    if not pattern or not s:
        return True
    return re.search(pattern=pattern, string=s) is not None


def get_file_paths(path, pattern):
    if not path:
        raise FileNotFoundError(f"Jsonl conversation log path does not exist: {path}")
    if os.path.isdir(path):
        file_paths = [join(path, f) for f in listdir(path) if isfile(join(path, f)) and matches(pattern=pattern, s=f)]
    else:
        file_paths = [path]
    return file_paths

def parse(path: str, pattern: str=None) -> Dict[str, List[str]]:
    data = {}
    for jsonl_path in get_file_paths(path=path, pattern=pattern):
        k = os.path.basename(jsonl_path).replace(".jsonl", "")
        v = read_jsonl(jsonl_path)
        data[k] = v
    return data

def parse_for_score(path: str, pattern: str=None) -> Dict[str, List[str]]:
    data = {}
    for jsonl_path in get_file_paths(path=path, pattern=pattern):
        k = os.path.basename(jsonl_path).replace(".jsonl", "")
        v = parse_one_conversation_log(jsonl_path=jsonl_path)
        data[k] = v
    return data


def join(glob_pattern: str, outpath: str):
    dataframes = []
    for filename in glob(glob_pattern):
        classification_prob = float(re.search(pattern="clarification_prob=([0-9.]+)", string=filename).group(1))
        task_type = re.search(pattern="task_type=([a-zA-Z]+)", string=filename).group(1)
        df = pd.read_json(filename, orient="records", lines=True)
        df["classification_prob"] = classification_prob
        df["task_type"] = task_type
        # rename all columns to add classification_prob except idx
        df.columns = [f"{c}_{classification_prob}" if c != "idx" else c for c in df.columns]

        dataframes.append(df)
    
    print(f"Joining {len(dataframes)} dataframes")
    # inner join on task_idx 
    reduced_df = reduce(lambda x, y: pd.merge(x, y, on="idx"), dataframes)
    print(reduced_df.head())
    reduced_df.to_csv(outpath, index=False, sep="\t")
    print(f"Saved to {outpath}")


def read_jsonl(file_path):
    output = []
    with open(file_path, 'r') as open_file:
        for line in open_file:
            output.append(json.loads(line))
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to conversation logs", type=str)
    parser.add_argument("--pattern", help="when path is a directory e.g., logs/ then select the files that "
                                               "match the pattern. Currently only support substring match.")
    parser.add_argument("--out_path", help="to store the output", type=str, default=tempfile.NamedTemporaryFile(delete=False, suffix = '.tsv').name)
    parser.add_argument("--parse", help="parse the conversation logs", action="store_true")
    parser.add_argument("--join", help="join multiple conversation logs", action="store_true")
    args = parser.parse_args()
    if args.parse:
        data = parse(path=args.path, pattern=args.pattern)
        df = pd.DataFrame(data=data)
        df.to_csv(args.out_path, sep="\t")
        print(f"\n\nOutput is in {args.out_path}")
    
    elif args.join:
        join(glob_pattern=args.pattern, outpath=args.out_path)
