import json
from data.io import read_fasta
from data.vocab import load_esm_alphabet
from data.datasets import TemporalFastaDataset
from data.data_modules.base_dm import ProteinGISAIDDataModule
from data.data_modules import register_dm
from copy import deepcopy
from utils.args import str2bool
import numpy as np
from collections import defaultdict


@register_dm("lm_weighted")
class ProteinLMWeightedDataModule(ProteinGISAIDDataModule):
    def __init__(self, args, vocab=None):
        super().__init__(args)
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = load_esm_alphabet(self.args.vocab, self.args.mol_type)

    def build_predict_datasets(self, properties, properties_dict):
        if self.args.set_data_properties is not None:
            set_data_properties = json.loads(self.args.set_data_properties)
            self.pred_datasets = [self.build_predicting_set(None, properties, properties_dict, set_data_properties=set_data_properties)]
        else:
            self.pred_datasets = [self.build_predicting_set(None, properties, properties_dict)]
        
    def build_predicting_set(self, pred_data_path, properties, *args, **argv):
        set_data_properties = argv.get("set_data_properties", {})
        fake_fasta = []
        for time in range(self.args.min_testing_time, self.args.max_testing_time + 1):
            desc = "time_bin=%d|freq=1.0" % (time)
            if len(set_data_properties) > 0:
                desc = desc + "|" + "|".join(["%s=%s" % (k, v) for k, v in set_data_properties.items()])
            fake_fasta.extend([("gen%d" % (i + len(fake_fasta)), "", "gen%d %s" % (i + len(fake_fasta), desc)) for i in range(self.args.generation_seq_number)])
        pred_set = TemporalFastaDataset(fake_fasta, self.vocab, get_time_method="kw", properties=properties)
        return pred_set


    def build_testing_set(self, test_data_path, properties):
        if self.args.set_data_properties is not None:
            set_data_properties = json.loads(self.args.set_data_properties)
        else:
            set_data_properties = {}
        test_set = TemporalFastaDataset(read_fasta(test_data_path), self.vocab, get_time_method="kw", properties=properties)
                
        extended_test_set = []
        if self.args.min_testing_time != -1 and self.args.max_testing_time != -1:
            for item in test_set:
                for key in set_data_properties:
                    item[key] = set_data_properties[key]
                
                for time in range(self.args.min_testing_time, self.args.max_testing_time + 1):
                    new_item = deepcopy(item)
                    new_item["src_time"] = time
                    extended_test_set.append(new_item)
        else:
            return test_set
        return extended_test_set
         
@register_dm("lm_weighted_location")
class ProteinLMWeightedLocationDataModule(ProteinLMWeightedDataModule):
    def __init__(self, args, vocab=None):
        super().__init__(args, vocab)

        if args.continent_to_country_mapping_file is not None:
            self.continent_to_country = json.load(open(args.continent_to_country_mapping_file))
            self.country_to_continent = {}
            for continent, countries in self.continent_to_country:
                if self.args.add_other_countries:
                    countries.append("other_countries")      
                for country in countries:
                    self.country_to_continent[country] = continent
        else:
            self.continent_to_country, self.country_to_continent = None, None
        
    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super(ProteinLMWeightedDataModule, cls).add_argparse_args(parent_parser)
        parent_parser.add_argument('--continent_to_country_mapping_file', type=str, default=None, help="The file mapping each country to a continent.")
        parent_parser.add_argument('--shuffle_property', type=str2bool, default="false")
        parent_parser.add_argument('--remap_continent', type=str2bool, default="false")
        parent_parser.add_argument('--add_other_countries', type=str2bool, default="false")
        return parent_parser

    def build_property_dict(self, full_dataset, properties_dict):
        prop_lists = defaultdict(set)
        for i in range(len(full_dataset)):
            data = full_dataset[i]
            for prop in self.args.data_properties:
                prop_lists[prop].add(data.get(prop, None))
        
        for prop in prop_lists:
            prop_list = list(prop_lists[prop])           
            
            if not (len(prop_list) == 1 and prop_list[0] is None): # nothing is included
                if None in prop_list:
                    prop_list.remove(None)
                prop_list.sort()

                if prop == "continent":
                    prop_list = [x[0] for x in self.continent_to_country]
                    prop_dict = {v: idx for idx, v in enumerate(prop_list)}
                elif prop == "country":
                    if "continent" in prop_lists:
                        prop_list = self.continent_to_country
                        prop_dict = {}
                        for _continent, _countries in self.continent_to_country:
                            prop_dict = {**prop_dict, **{v: idx for idx, v in enumerate(_countries)}}
                        setattr(self.args, "contient2country", self.continent_to_country)
                    else:
                        prop_list = [y for x in self.continent_to_country for y in x[1]]
                        prop_list.sort()
                        prop_dict = {v: idx for idx, v in enumerate(prop_list)}
                elif prop == "location" and "continent" in prop_lists:
                    prop_list = self.continent_to_country
                    prop_dict = {}
                    for _continent, _countries in self.continent_to_country:
                        prop_dict = {**prop_dict, **{"%s/%s" % (_continent, v): idx for idx, v in enumerate(_countries)}}
                    setattr(self.args, "contient2country", self.continent_to_country)                    
                    prop_dict = CoutryVocab(prop_dict)
                else: # !!! This is used for the continent level
                    prop_dict = {v: idx for idx, v in enumerate(prop_list)}
                # else:
                #     prop_dict = {v: idx for idx, v in enumerate(prop_list)}
                setattr(self.args, "%s_dict" % prop, prop_dict)
                setattr(self.args, "%s_list" % prop, prop_list)
                
                properties_dict[prop] = prop_dict
        
    def remap_continent(self, dataset):
        new_dataset = []
        for data in dataset:            
            if "location" in data:
                location = data["location"]
                if location.split("/")[1] not in self.country_to_continent:
                    new_continent = self.country_to_continent["other_countries"]
                    new_country = "other_countries"
                else:
                    new_continent = self.country_to_continent[location.split("/")[1]]
                    new_country = location.split("/")[1]

                new_location = "%s/%s" % (new_continent, new_country)
                data["location"] = new_location
                if "continent" in data:
                    data["continent"] = new_continent # self.country_to_continent[location.split("/")[1]]
                new_dataset.append(data)
            elif "country" in data:
                location = data["country"]
                data["country"] = location
                
                if "continent" in data:
                    if location not in self.country_to_continent:
                        new_continent = self.country_to_continent["other_countries"]
                    else:
                        new_continent = self.country_to_continent[location]
                    data["continent"] = new_continent # self.country_to_continent[location]
                # print(data)
                new_dataset.append(data)
            else:
                new_dataset.append(data)
        # print(new_dataset[0])
        return new_dataset

    def build_training_set(self, properties, properties_dict):
        datasets = []
        for data_path in self.args.data_path:
            dataset = read_fasta(data_path)
            datasets.extend(dataset)
        
        full_dataset = TemporalFastaDataset(datasets, self.vocab, 
            get_time_method="kw", 
            properties=properties)
        self.build_property_dict(full_dataset, properties_dict)
        
        if self.args.remap_continent:
            full_dataset = self.remap_continent(full_dataset)

        if self.args.shuffle_property:
            shuffle_index = np.arange(len(full_dataset))
            np.random.shuffle(shuffle_index)
            data_property = self.args.data_properties[0]
            labels = [x[data_property] for x in full_dataset]
            shuffle_labels = [labels[idx] for idx in shuffle_index]
            full_dataset_new = []
            for i, x in enumerate(full_dataset):
                x[data_property] = shuffle_labels[i]
                full_dataset_new.append(x)
            full_dataset = full_dataset_new
                
        return full_dataset

    # def calc_total_sample_count(self, train_set):
    #     # If loss weighted by count
    #     # If loss weighted by time?
    #     total_sample_count = 0
    #     for data in train_set:
    #         total_sample_count += round(data["bin_size"] * data["freq"]) 

    #     logging.info("total_sample_count: %d" % total_sample_count)
    #     # self.total_sample_weight = []
    #     return total_sample_count

    def build_test_datasets(self, properties, properties_dict, *args, **argv):
        self.test_datasets = []
        for test_data_path in self.args.test_data_paths:
            test_dataset = self.build_testing_set(test_data_path, properties)
            if self.args.remap_continent:
                test_dataset = self.remap_continent(test_dataset)
            self.test_datasets.append(test_dataset)

    def build_predicting_set(self, pred_data_path, properties, *args, **argv):
        set_data_properties = argv.get("set_data_properties", {})
        fake_fasta = []
        for time in range(self.args.min_testing_time, self.args.max_testing_time + 1):
            desc = "time_bin=%d|freq=1.0" % (time)
            if len(set_data_properties) > 0:
                desc = desc + "|" + "|".join(["%s=%s" % (k, v) for k, v in set_data_properties.items()])
            fake_fasta.extend([("gen%d" % (i + len(fake_fasta)), "", "gen%d %s" % (i + len(fake_fasta), desc)) for i in range(self.args.generation_seq_number)])
        pred_set = TemporalFastaDataset(fake_fasta, self.vocab, get_time_method="kw", properties=properties)
        if self.args.remap_continent:
            pred_set = self.remap_continent(pred_set)
        return pred_set
    
    # def train_dataloader(self, ):
    #     sampler = None
    #     shuffle = True
    #     train_loader = DataLoader(
    #         self.train_dataset, 
    #         batch_size=self.args.batch_size, 
    #         shuffle=shuffle, 
    #         pin_memory=self.args.pin_memory, 
    #         num_workers=self.args.num_workers, 
    #         persistent_workers=self.args.persistent_workers,
    #         collate_fn=self.collate_fn,
    #         sampler=sampler
    #     )
    #     return train_loader

    # def val_dataloader(self):
    #     val_loader = DataLoader(
    #         self.val_dataset, 
    #         batch_size=self.args.batch_size, 
    #         shuffle=False, 
    #         num_workers=self.args.num_workers, 
    #         persistent_workers=self.args.persistent_workers,
    #         collate_fn=self.collate_fn
    #     )
    #     return val_loader

    # def test_dataloader(self, test_datasets=None, repeat_size=1, load_history=False, batched=False):
    #     test_loaders = []
    #     if test_datasets is None:
    #         test_datasets = self.test_datasets
            
    #     for test_dataset in test_datasets:
    #         test_loaders.append(
    #         DataLoader(
    #         test_dataset, 
    #         batch_size=self.args.batch_size, 
    #         shuffle=False, 
    #         pin_memory=self.args.pin_memory, 
    #         num_workers=self.args.num_workers, 
    #         persistent_workers=self.args.persistent_workers,
    #         collate_fn=self.collate_fn
    #         ))
    #     return test_loaders
    
    # def predict_dataloader(self,):
    #     pred_loaders = []
    #     for pred_dataset in self.pred_datasets:
    #         pred_loaders.append(
    #         DataLoader(
    #         pred_dataset, 
    #         batch_size=self.args.batch_size, 
    #         shuffle=False, 
    #         pin_memory=self.args.pin_memory, 
    #         num_workers=self.args.num_workers, 
    #         persistent_workers=self.args.persistent_workers,
    #         collate_fn=self.collate_fn
    #         ))
    #     return pred_loaders
