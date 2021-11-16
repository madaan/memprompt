import logging
import Levenshtein as lev
import re
import json

logging.basicConfig(level=logging.INFO)


class ClosestMatchMemory(object):
    def __init__(self, key_process_func: callable, min_sim_threshold: int = 6):
        """A memory that supports fuzzy matching of keys. The fuzzy matching is done by using levenshtein distance.

        Args:
            key_process_func (callable): A string --> string function, that is used to parse the keys before they
            are used to query or added.
            min_sim_threshold (int, optional): [description]. Defaults to 6.
        """
        self.memory = dict()
        self.min_sim_threshold = min_sim_threshold
        self.key_process_func = key_process_func
        self.min_sim_threshold = min_sim_threshold

    def parse_key(self, key: str) -> str:
        if self.key_process_func:
            return self.key_process_func(key)
        else:
            return key

    
    def __setitem__(self, key, value):
        key = self.parse_key(key)
        self.memory[key] = value

    def get_closest(self, key, return_score: bool = True):
        if len(self) == 0:
            return None, None, None
        key = self.parse_key(key)
        closest_key, closest_key_score = self._find_closest_key(key)
        if closest_key:
            if return_score:
                return closest_key, self[closest_key], closest_key_score
        else:
            return None, None, None
    

    def _find_closest_key(self, word):
        # find the key in self.memory that is closest to the word in terms of levenshtein distance
        min_dist = self.min_sim_threshold
        logging.debug("Finding closest key for word: {}".format(word))
        closest_key = None
        for key in self.memory:
            dist = lev.distance(word, key)
            logging.debug("Distance between {} and {} is {}".format(word, key, dist))
            if dist < min_dist:
                min_dist = dist
                closest_key = key
        if closest_key:
            return closest_key, min_dist
        else:
            return None, min_dist

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]

    def __str__(self) -> str:
        return " || ".join([f"{k}: {v}" for (k, v) in self.memory.items()])
        

def get_all_stopwords():
    from nltk.corpus import stopwords
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    all_stopwords = nlp.Defaults.stop_words
    sw = set(stopwords.words("english")).union(set(all_stopwords))
    return sw


if __name__ == "__main__":
    from nltk.corpus import stopwords

    sw = set()
    # write unit tests for ClosestMatchMemory
    def _process_key(key):
        key = re.sub("<.*>", "", key).lower().strip()
        # remove double spaces
        key = re.sub(" +", " ", key)
        # remove the punctuation
        key = re.sub("[^a-zA-Z0-9 ]", "", key)
        # remove the stop words
        key = " ".join([w for w in key.split() if w not in sw])
        return key

    memory = ClosestMatchMemory(lambda x: _process_key(x), min_sim_threshold=50)
    memory["hello"] = "world"
    memory["HELLO"] = "WORLD"
    memory["HELLO WORLD"] = "WORLD"
    memory["what rings like < papaya > ?"] = "clarification | rings like means homonym"
    memory[
        "use < carrot > in a sentence"
    ] = "clarification | use means give a sentence with the word"
    memory["< dice > ka matlab kya hai"] = "clarification | matlab means define"

    # pretty print the memory
    for key, value in memory.memory.items():
        print(f"{key}: {value}")

    # now check if the closest match is found
    assert memory.get_closest("hello") == "WORLD"
    assert memory.get_closest("HELLO") == "WORLD"
    assert memory.get_closest("HELLO WORLD") == "WORLD"
    assert memory.get_closest("HELLO WORLD!") == "WORLD"
    assert (
        memory.get_closest("what rings like < tomato > ?")
        == "clarification | rings like means homonym"
    )
    assert (
        memory.get_closest("use < cake > in a sentence")
        == "clarification | use means give a sentence with the word"
    )
    assert (
        memory.get_closest("< apple > ka matlab batana ") == "clarification | matlab means define"
    ), f"{memory.get_closest('< apple > ka matlab batana ')} != clarification | matlab means define"
    print("All tests passed")
