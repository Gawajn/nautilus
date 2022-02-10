import os
from collections import namedtuple
from itertools import groupby

import numpy as np
from ctc_decoder import BKTree, best_path, probability, loss
from torch.nn import CTCLoss

from symspellpy import SymSpell


class CharReplacementEngine:
    def __init__(self, path_to_ruleset=None):
        self.ruleSetDict = {}
        self.ruleSetDict["ſ"] = 's'
        self.ruleSetDict["ꝛ"] = 'r'

    def replace(self, replacement_string: str):
        for x in self.ruleSetDict.keys():
            replacement_string = replacement_string.replace(x, self.ruleSetDict[x])
        return replacement_string

import json
import os
from datetime import datetime
from typing import List

from dataclasses import dataclass

from dataclasses_json import dataclass_json



@dataclass_json
@dataclass
class WordFrequency:
    word: str
    frequency: int
    hyphenated: str


@dataclass_json
@dataclass
class WordFrequencyDict:
    freq_list: List[WordFrequency]

    #def get_n_most_similar_words(self, word, n=5):
    #    probs = {}
     #   total = sum([x.frequency for x in self.freq_list])
     #   for k in self.freq_list:
     #       probs[k.word] = k.frequency / total
     #   if word in probs:
     #       return "yay"
     #   else:
     #       similarities = [1 - (textdistance.Jaccard(qval=2).distance(v, input_word)) for v in word_freq_dict.keys()]




class DatabaseDictionary:
    def __init__(self, b_id: str = None, name: str = '', created: datetime = datetime.now(),
                 dictionary=None
                 ):
        if dictionary is None:
            dictionary = []
        self.b_id = b_id
        self.name: str = name
        self.created: datetime = created
        self.dictionary: WordFrequencyDict = dictionary

    def to_file(self, id):
        self.b_id = id
        s = self.to_json()
        with open(id.local_path('book_dictionary.json'), 'w') as f:
            js = json.dumps(s, indent=2)
            f.write(js)

    @staticmethod
    def from_json(json: dict):
        return DatabaseDictionary(
            name=json.get('name', ""),
            created=datetime.fromisoformat(json.get('created', datetime.now().isoformat())),
            dictionary=WordFrequencyDict.from_dict(json.get('dictionary'))

        )

    def to_json(self):
        return {
            "name": self.name,
            "created": self.created.isoformat(),
            "dictionary": self.dictionary.to_dict() if self.dictionary else []
        }
    def to_frequent_list(self):
        pass

class DictionaryCorrector(SymSpell):
    def __init__(self, max_dictionary_edit_distance=2, prefix_length=7,
                 count_threshold=1):
        super().__init__(max_dictionary_edit_distance=max_dictionary_edit_distance, prefix_length=prefix_length,
                         count_threshold=count_threshold)
        #self.sym_spell = SymSpell()
        self.book_id = None

    def segmentate_correct_text(self, text, edit_distance=2):
        sentence = self.word_segmentation(text, edit_distance)
        return sentence

    def load_dict(self, name):

        path = "default_dictionary.json"
        with open(path) as f:
            d = DatabaseDictionary.from_json(json.load(f))
        for entry in d.dictionary.freq_list:
            print(entry)
            self.create_dictionary_entry(entry.word, entry.frequency)
        path = 'bigram_default_dictionary.txt'
        loaded = self.load_bigram_dictionary(path, 0, 2)
        print(loaded)
        pass


