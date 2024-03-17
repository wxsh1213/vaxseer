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

class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(self.args)
    
    @classmethod
    def add_argparse_args(cls, parent_parser):
        # group = parent_parser.add_argument_group('ProteinDataModule')
        
        # New
        parent_parser.add_argument('--pin_memory', type=str2bool, default="true")
        parent_parser.add_argument('--pre_tokenize', type=str2bool, default="false", help="Tokenize sequences in the initialization?")
        parent_parser.add_argument('--cache_token', type=str2bool, default="false", help="Saving the tokenization!")
        parent_parser.add_argument('--data_path', type=str, default=None, nargs="+")
        parent_parser.add_argument('--test_data_paths', nargs="+", type=str, default=None) # , ])
        parent_parser.add_argument('--disable_autobatch', type=str2bool, default="false")
        parent_parser.add_argument('--max_position_embeddings', type=int, default=1024) # TODO: put it here or in the model part?
        # New

        # parent_parser.add_argument('--data_dir', type=str, default="")
        parent_parser.add_argument('--vocab', type=str, default="", help="If not specified, will be modified according to the model_name_or_path.")
        parent_parser.add_argument('--valid_size', type=float, default=0.1)
        parent_parser.add_argument('--batch_size', type=int, default=32)
        parent_parser.add_argument('--num_workers', type=int, default=0)
        parent_parser.add_argument('--persistent_workers', type=str2bool, default=False)
        # For testing: 
        # parent_parser.add_argument('--test_data_dirs', nargs="+", default=['/data/rsg/nlp/wenxian/esm/data/cov_spike/align_2022_test_top1'], type=str)
        parent_parser.add_argument('--pred_data_paths', nargs="+", default="", type=str)

        parent_parser.add_argument('--predict_src_file', default="", type=str)
        parent_parser.add_argument('--predict_tgt_file', default="", type=str)
        parent_parser.add_argument('--source_sample_num', default=1, type=int, )
        parent_parser.add_argument('--predict_sample_num', default=1, type=int, )

        # 
        parent_parser.add_argument('--mol_type', type=str, default="protein", choices=["dna_codon", "rna_codon", "protein"])
        
        # group.add_argument('--test_target_path',default='/data/rsg/nlp/wenxian/esm/data/cov_spike/test_2022/msa/test_align_2022-01.query_2022-02.target.fasta', type=str, help="Path of the target.")
        # group.add_argument('--generation_order', default=1, type=int)
        # group.add_argument('--requires_alignment', action='store_true', help="If the target is required to be aligned.")
        return parent_parser

class PairwiseProteinDataModule(ProteinDataModule):
    def __init__(self, args):
        super().__init__(args)
    
    def prepare_data(self):
        # download data, etc...
        # only called on 1 GPU/TPU in distributed
        # Avoid asign any variables here.
        pass

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        # Assign train/val datasets for use in dataloaders
        
        # Load datasets
        self.vocab = load_esm_alphabet(self.args.vocab, self.args.mol_type)
        if stage == "fit" or stage is None:
            src_fasta_path = os.path.join(self.args.data_dir, "train.src.fasta")
            tgt_fasta_path = os.path.join(self.args.data_dir, "train.tgt.fasta")
            src_dataset = read_fasta(src_fasta_path)
            tgt_dataset = read_fasta(tgt_fasta_path)
            full_dataset = TemporalPairwiseFastaDataset(src_dataset, tgt_dataset, self.vocab)

            valid_size = int(len(full_dataset) * self.args.valid_size)
            train_size = len(full_dataset) - valid_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, valid_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_datasets = []
            for test_data_dir in self.args.test_data_dirs:
                src_fasta_path = os.path.join(test_data_dir, "test.src.fasta")
                tgt_fasta_path = os.path.join(test_data_dir, "test.tgt.fasta")
                src_dataset = read_fasta(src_fasta_path)
                tgt_dataset = read_fasta(tgt_fasta_path)
                self.test_datasets.append(TemporalPairwiseFastaDataset(src_dataset, tgt_dataset, self.vocab, get_time_method="kw"))

        if stage == "predict":
            self.predict_dataset = []
            src_dataset = read_fasta(self.args.predict_src_file)
            self.predict_dataset = TemporalFastaDataset(src_dataset, self.vocab)
            # self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)
        
        if stage == "test_population":
            # Randomly choose a prior distribution:
            src_dataset = read_fasta(self.args.predict_src_file)
            tgt_dataset = read_fasta(self.args.predict_tgt_file)
            self.test_population_dataset = TemporalUnpairedFastaDataset(src_dataset, tgt_dataset, self.vocab, source_sample_num=self.args.source_sample_num)
            # self.predict_tgt_dataset = TemporalFastaDataset(tgt_dataset, self.vocab)
            print(len(self.test_population_dataset))
            print(self.test_population_dataset[0])

            # self.test_datasets = []
            # for test_data_dir in self.args.test_data_dirs:
            #     src_fasta_path = os.path.join(test_data_dir, "test.src.fasta")
            #     tgt_fasta_path = os.path.join(test_data_dir, "test.tgt.fasta")
            #     src_dataset = read_fasta(src_fasta_path)
            #     tgt_dataset = read_fasta(tgt_fasta_path)
            #     self.test_datasets.append(TemporalPairwiseFastaDataset(src_dataset, tgt_dataset, self.vocab, get_time_method="kw"))

    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     # Do something before the batch, like da or add some noise?
    #     batch['x'] = transforms(batch['x'])
    #     return batch

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            # pin_memory=True, 
            num_workers=self.args.num_workers, 
            persistent_workers=self.args.persistent_workers,
            collate_fn=partial(customized_collate_func, batch_converter=self.vocab.get_batch_converter())
        )
        
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            persistent_workers=self.args.persistent_workers,
            collate_fn=partial(customized_collate_func, batch_converter=self.vocab.get_batch_converter())
        )
        return val_loader

    def test_dataloader(self):
        test_loaders = []
        # test_population_dataset
        for test_dataset in  [self.test_population_dataset]: # self.test_datasets:
            test_loaders.append(
                DataLoader(
                test_dataset, 
                batch_size=self.args.batch_size, 
                shuffle=False, 
                num_workers=self.args.num_workers, 
                persistent_workers=self.args.persistent_workers,
                collate_fn=partial(customized_collate_func, batch_converter=self.vocab.get_batch_converter(), aligned=False)
            ))
        return test_loaders
    
    def predict_dataloader(self, source="src", remove_gaps=True, batch_size=None):
        predict_loader = DataLoader(
            self.predict_src_dataset if source == "src" else self.predict_tgt_dataset, # Monolingual acturally... 
            batch_size=batch_size if batch_size is not None else self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            persistent_workers=self.args.persistent_workers,
            collate_fn=partial(customized_collate_func, batch_converter=self.vocab.get_batch_converter(), remove_gaps=remove_gaps)
        )
        return predict_loader

    def teardown(self, stage, *args, **kwargs):
        # clean up after fit or test
        # called on every process in DDP
        pass

class ProteinGISAIDDataModule(ProteinDataModule):
    def __init__(self, args, vocab=None):
        super().__init__(args)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super(ProteinGISAIDDataModule, cls).add_argparse_args(parent_parser)
        parent_parser.add_argument('--generation_seq_number', type=int, default=1)
        # parent_parser.add_argument('--continent_to_country_mapping_file', type=str, default="data/data_modules/continent2countries_minCnt1000.json")
        # parent_parser.add_argument('--split_valid_set_by_time', type=str2bool, default="false")
        parent_parser.add_argument('--data_properties', nargs="+", type=str, default=[], help="What kind of information is stored in dataset.")
        parent_parser.add_argument('--set_data_properties', type=str, default=None, help="Set up the property values when generating.")
        # parent_parser.add_argument('--debug_mess_data_property', type=str, default=None)
        # parent_parser.add_argument('--property_weighted_random_sampler', type=str2bool, default="false")
        # parent_parser.add_argument('--remap_continent', type=str2bool, default="false")
        # parent_parser.add_argument('--hierarchical_properties', nargs="+", type=str, default=None)
        # parent_parser.add_argument('--load_location', type=str2bool, default="true", help="Load location information.")
        # parent_parser.add_argument('--load_lineage', type=str2bool, default="true", help="Load lineage information.")
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
                print(prop)
                properties_dict[prop] = getattr(model_config, "%s_dict" % prop)
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
