# -*- coding: utf-8 -*-
# 把数据处理的部分放在一个类里，这样更清晰
# 而且可以把数据处理的部分放在GPU上，这样更快
# 而且可以异步处理数据，加速训练
import logging
import os
import json
import pickle
import random
import dataclasses
import ast
import logging
import pandas as pd
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset
from tqdm import tqdm
from nltk.corpus import wordnet as wn

@dataclass(frozen=True)
class InputFeature:
    """
    A single set of features of data.
    """
    input_ids: List[int]
    input_attention_mask: List[int]
    target_ids: List[int]
    target_attention_mask: List[int]
    word_ids: List[int]
    
    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

class WSDDataset(Dataset):
    """
    Self-defined WSD Dataset class.
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,
        data_path,
        data_partition,
        tokenizer,
        cache_dir=None,
        is_test=False,
    ):
        self.data_partition = data_partition
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.is_test = is_test
        
        self.instances = []
        self._cache_instances(data_path)
    
    def _cache_instances(self, data_path):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        signature = "{}_cache.pkl".format(self.data_partition)
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, signature)
        else:
            cache_dir = os.mkdir("caches")
            cache_path = os.path.join(cache_dir, signature)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                logging.info ("Loading cached instances from {}".format(cache_path))
                self.instances = pickle.load(f)
        else:          
            logging.info ("Loading raw data from {}".format(data_path))
            all_samples = []
            data_db = pd.read_csv(data_path)
            for id, sentence, sense_keys, glosses, targets,words in tqdm(zip(data_db["id"], data_db["sentence"], data_db["sense_keys"], data_db["glosses"], data_db["targets"],data_db["word"])): 
                data_sample = {
                    "id": id,
                    "sentence": sentence,
                    "sense_keys": ast.literal_eval(sense_keys),
                    "glosses": ast.literal_eval(glosses),
                    "targets": ast.literal_eval(targets),
                    "words":words
                }
                all_samples.append(data_sample)
            with open(cache_path.replace(".pkl", ".json"), 'w') as f:
                json.dump(all_samples, f, indent=4)
            
            logging.info ("Creating cache instances {}".format(signature))
            
            for sample in tqdm(all_samples):
                input_ids, input_attn_mask, target_ids, target_attn_mask,words_ids = self._parse_input(sample)
                inputs = {
                    "input_ids": input_ids,
                    "input_attention_mask": input_attn_mask,
                    "target_ids": target_ids,
                    "target_attention_mask": target_attn_mask,
                    "words_ids":words_ids,
                }
                feature = InputFeature(**inputs)
                self.instances.append(feature)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info ("Total of {} instances were cached.".format(len(self.instances)))
     
    def _parse_input(self, sample: dict):
        input_ids = self.tokenizer(sample["sentence"], padding=True, truncation=True, return_tensors="pt")["input_ids"][0].tolist()        
        input_attn_mask = self.tokenizer(sample["sentence"], padding=True, truncation=True, return_tensors="pt")["attention_mask"][0].tolist()
        target_ids = self.tokenizer(sample["glosses"][sample["targets"][0]], padding=True, truncation=True, return_tensors="pt")["input_ids"][0].tolist()
        target_attn_mask = self.tokenizer(sample["glosses"][sample["targets"][0]], padding=True, truncation=True, return_tensors="pt")["attention_mask"][0].tolist()
        words_ids = self.tokenizer(sample["words"], padding=True, truncation=True, return_tensors="pt")["input_ids"][0].tolist()
        return input_ids, input_attn_mask, target_ids, target_attn_mask,words_ids

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]

