import json
from data.io import read_fasta
from data.vocab import load_esm_alphabet
from data.datasets import TemporalFastaDataset
from data.data_modules.base_dm import ProteinGISAIDDataModule
from data.data_modules import register_dm
from copy import deepcopy

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
        if self.args.remap_continent:
            pred_set = self.remap_continent(pred_set)
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
         
