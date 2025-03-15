from collections import defaultdict
import pytorch_lightning as pl
import os
from functools import partial
from data.io import read_fasta
from data.vocab import load_esm_alphabet
from data.datasets import TemporalFastaDataset
from data.utils import default_lm_collate_func
from torch.utils.data import DataLoader, random_split
from utils.args import str2bool
from copy import deepcopy
import json
from collections.abc import Mapping

class CoutryVocab(Mapping):
    def __init__(self, vocab) -> None:
        super().__init__()
        self.vocab = vocab
    
    def __iter__(self):
        for tok in self.vocab:
            yield tok

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, tok):
        assert len(tok.split("/")) >= 2, "Cannot find the continent and country information from %s" % tok
        continent, country = tok.split("/")
        if tok in self.vocab:
            return self.vocab[tok]
        else:
            return self.vocab["%s/other_countries" % continent]
    
    def get(self, key):
        self.__getitem__(key)

# deprecated:
class GeneralLocationVocab(Mapping):
    def __init__(self, vocab) -> None:
        super().__init__()
        self.vocab = vocab
    
    def __iter__(self):
        for tok in self.vocab:
            yield tok

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, tok):
        if len(tok.split("/")) <= 1:
            return self.vocab[tok]
        assert len(tok.split("/")) == 2
        continent, country = tok.split("/")
        if tok in self.vocab:
            return self.vocab[tok]
        else:
            return self.vocab["%s/other_countries" % continent]
    
    def get(self, key):
        self.__getitem__(key)


class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(self.args)
    
    @classmethod
    def add_argparse_args(cls, parent_parser):        
        parent_parser.add_argument('--pin_memory', type=str2bool, default="true")
        parent_parser.add_argument('--pre_tokenize', type=str2bool, default="false", help="Tokenize sequences in the initialization?")
        parent_parser.add_argument('--cache_token', type=str2bool, default="false", help="Saving the tokenization!")
        parent_parser.add_argument('--data_path', type=str, default=None, nargs="+")
        parent_parser.add_argument('--test_data_paths', nargs="+", type=str, default=None) # , ])
        parent_parser.add_argument('--disable_autobatch', type=str2bool, default="false")
        parent_parser.add_argument('--max_position_embeddings', type=int, default=1024) # TODO: put it here or in the model part?

        parent_parser.add_argument('--vocab', type=str, default="", help="If not specified, will be modified according to the model_name_or_path.")
        parent_parser.add_argument('--valid_size', type=float, default=0.1)
        parent_parser.add_argument('--batch_size', type=int, default=32)
        parent_parser.add_argument('--num_workers', type=int, default=0)
        parent_parser.add_argument('--persistent_workers', type=str2bool, default=False)
        parent_parser.add_argument('--pred_data_paths', nargs="+", default="", type=str)

        parent_parser.add_argument('--source_sample_num', default=1, type=int, )
        parent_parser.add_argument('--predict_sample_num', default=1, type=int, )

        parent_parser.add_argument('--mol_type', type=str, default="protein", choices=["dna_codon", "rna_codon", "protein"])

        return parent_parser

class ProteinGISAIDDataModule(ProteinDataModule):
    def __init__(self, args, vocab=None):
        super().__init__(args)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super(ProteinGISAIDDataModule, cls).add_argparse_args(parent_parser)
        parent_parser.add_argument('--generation_seq_number', type=int, default=1)
        parent_parser.add_argument('--data_properties', nargs="+", type=str, default=[], help="What kind of information is stored in dataset.")
        parent_parser.add_argument('--set_data_properties', type=str, default=None, help="Set up the property values when generating.")
        return parent_parser
    
    def reset_testing_time(self, test_set, time_key="src_time"):
        if self.args.min_testing_time != -1 and self.args.max_testing_time != -1:
            extended_test_set = []
            for item in test_set:
                for time in range(self.args.min_testing_time, self.args.max_testing_time + 1):
                    new_item = deepcopy(item)
                    new_item[time_key] = time
                    extended_test_set.append(new_item)
            return extended_test_set
        else:
            return test_set

    def build_property_dict(self, full_dataset, properties_dict):
        prop_lists = defaultdict(set)
        
        for data in full_dataset:
            for prop in self.args.data_properties:
                prop_lists[prop].add(data.get(prop, None)) # None

        for prop in prop_lists:
            prop_list = list(prop_lists[prop])        
            prop_list.sort()
            if not (len(prop_list) == 1 and prop_list[0] is None): # nothing is included
                prop_dict = {v: idx for idx, v in enumerate(prop_list)}
                setattr(self.args, "%s_dict" % prop, prop_dict)
                setattr(self.args, "%s_list" % prop, prop_list)
                properties_dict[prop] = prop_dict

    def build_training_set(self, properties, properties_dict):
        datasets = []
        for data_path in self.args.data_path:
            dataset = read_fasta(data_path)
            datasets.extend(dataset)
        
        full_dataset = TemporalFastaDataset(datasets, self.vocab, get_time_method="kw", properties=properties) # 
        self.build_property_dict(full_dataset, properties_dict)
        return full_dataset
    
    def build_testing_set(self, test_data_path, properties):
        test_set = TemporalFastaDataset(read_fasta(test_data_path), self.vocab, get_time_method="kw", properties=properties)
        return test_set

    def set_collate_func(self, properties_dict, *args, **kwargs):                    
        self.collate_fn = partial(default_lm_collate_func, \
                batch_converter=self.vocab.get_batch_converter(max_positions=self.args.max_position_embeddings), 
                padding_idx=self.vocab.pad(), properties_dict=properties_dict
            )

    def setup_properties(self, ):
        properties = ['time_bin', 'freq', 'bin_size']
        properties += self.args.data_properties
        return properties
        
    def load_properties_from_config(self, model_config, properties_dict):
        if model_config:
            for prop in getattr(model_config, "data_properties", []):
                print("Load property:", prop)
                properties_dict[prop] = getattr(model_config, "%s_dict" % prop)
                print(properties_dict[prop])
        return properties_dict

    def setup(self, stage, model_config=None):
        properties_dict = {}
        # Load datasets
        self.tokenizer = self.vocab # consistent with TransformerLM
        
        properties = self.setup_properties()

        if stage == "fit" or stage is None:
            full_dataset = self.build_training_set(properties, properties_dict)
            if len(full_dataset) == 2:
                self.train_dataset, self.val_dataset = full_dataset[0], full_dataset[1]
            else:
                valid_size = int(len(full_dataset) * self.args.valid_size)
                train_size = len(full_dataset) - valid_size
                self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, valid_size])

        if stage == "test" or stage == "predict":
            properties_dict = self.load_properties_from_config(model_config, properties_dict)

        if stage == "test":
            self.build_test_datasets(properties, properties_dict)
            
        if stage == "predict":
            self.build_predict_datasets(properties, properties_dict)
        
        self.set_collate_func(properties_dict, model_config)
    
    def build_predict_datasets(self, properties, properties_dict):
        self.pred_datasets = []
        for pred_data_path in self.args.pred_data_paths:
            pred_dataset = self.build_predicting_set(pred_data_path, properties, properties_dict)
            self.pred_datasets.append(pred_dataset)

    def build_test_datasets(self, properties, properties_dict, *args, **argv):
        self.test_datasets = []
        for test_data_path in self.args.test_data_paths:
            test_dataset = self.build_testing_set(test_data_path, properties)
            self.test_datasets.append(test_dataset)

    def build_predicting_set(self, pred_data_path, properties, *args, **argv):
        return TemporalFastaDataset(read_fasta(pred_data_path), self.vocab, get_time_method="kw", properties=properties)

    def train_dataloader(self, ):
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            pin_memory=self.args.pin_memory, 
            num_workers=self.args.num_workers, 
            persistent_workers=self.args.persistent_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            persistent_workers=self.args.persistent_workers,
            collate_fn=self.collate_fn
        )
        return val_loader

    def test_dataloader(self, test_datasets=None, repeat_size=1, load_history=False, batched=False):
        test_loaders = []
        if test_datasets is None:
            test_datasets = self.test_datasets
            
        for test_dataset in test_datasets:
            test_loaders.append(
            DataLoader(
            test_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            pin_memory=self.args.pin_memory, 
            num_workers=self.args.num_workers, 
            persistent_workers=self.args.persistent_workers,
            collate_fn=self.collate_fn
            ))
        return test_loaders
    
    def predict_dataloader(self,):
        pred_loaders = []
        for pred_dataset in self.pred_datasets:
            pred_loaders.append(
            DataLoader(
            pred_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            pin_memory=self.args.pin_memory, 
            num_workers=self.args.num_workers, 
            persistent_workers=self.args.persistent_workers,
            collate_fn=self.collate_fn
            ))
        return pred_loaders
