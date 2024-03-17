import pandas as pd
from collections import Counter
from data.io import read_fasta
from data.vocab import load_esm_alphabet
from data.datasets import PairwiseClassificationDataset, PairwiseAlnClassificationDataset
from data.data_modules.base_dm import ProteinGISAIDDataModule
from data.data_modules import register_dm
from utils.args import str2bool

@register_dm("hi_regression")
class PairwiseRegressionDataModule(ProteinGISAIDDataModule):
    def __init__(self, args, vocab=None):
        super().__init__(args)
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = load_esm_alphabet(self.args.vocab)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super(PairwiseRegressionDataModule, cls).add_argparse_args(parent_parser)
        # parent_parser.add_argument('--meta_data_path', type=str, default="/data/rsg/nlp/wenxian/esm/data/gisaid/flu/metadata.csv", help="Load location information.")
        # parent_parser.add_argument('--labels', nargs="+", type=str, default=None)
        # parent_parser.add_argument('--binary', type=str2bool, default="true")
        parent_parser.add_argument('--train_index_path', type=str, default=None)
        parent_parser.add_argument('--train_loss_weight_path', type=str, default=None)
        parent_parser.add_argument('--valid_index_path', type=str, default=None)
        parent_parser.add_argument('--valid_loss_weight_path', type=str, default=None)
        parent_parser.add_argument('--test_index_path', type=str, default=None)
        parent_parser.add_argument('--test_loss_weight_path', type=str, default=None)
        parent_parser.add_argument('--predict_index_path', type=str, default=None)
        parent_parser.add_argument('--category', type=str2bool, default="true")
        parent_parser.add_argument('--numerical', type=str2bool, default="false")
        parent_parser.add_argument('--numerical_interval', type=float, default=1.0)
        

        parent_parser.add_argument('--virus_id_col_name', type=str, default="virus")
        parent_parser.add_argument('--vaccine_id_col_name', type=str, default="reference")
        parent_parser.add_argument('--virus_seq_col_name', type=str, default="virus_seq")
        parent_parser.add_argument('--vaccine_seq_col_name', type=str, default="reference_seq")
        parent_parser.add_argument('--value_col_name', type=str, default="hi")

        parent_parser.add_argument('--prepend_special_token_for_vaccine', type=str, default=None, help="prepend special tokens for vaccine strains.")
        parent_parser.add_argument('--prepend_special_token_for_virus', type=str, default=None, help="prepend special tokens for vaccine strains.")
        return parent_parser
    
    def setup_properties(self, ):
        # return ["seq_label", "ref_seq_label"]
        return "label"
    
    def read_csv(self, path):
        data = []
        df = pd.read_csv(path)
        for virus_id, vaccine_id, virus_seq, vaccine_seq, value in zip(df[self.args.virus_id_col_name], df[self.args.vaccine_id_col_name], df[self.args.virus_seq_col_name], df[self.args.vaccine_seq_col_name], df[self.args.value_col_name]):                
            data.append((virus_id, vaccine_id, virus_seq, vaccine_seq, value))

        # data = []
        # with open(path) as csvfile:
        #     spamreader = csv.reader(csvfile)
        #     for i, row in enumerate(spamreader):
        #         if i == 0:
        #             headline = row
        #             continue
        #         data.append(row)
        return data

    def build_training_set(self, properties, properties_dict):      
        dataset = read_fasta(self.args.data_path)
        train_index_data = self.read_csv(self.args.train_index_path)
        train_dataset = PairwiseClassificationDataset(dataset, self.vocab, index_data=train_index_data, category=self.args.category)
        valid_index_data = self.read_csv(self.args.valid_index_path)
        valid_dataset = PairwiseClassificationDataset(dataset, self.vocab, index_data=valid_index_data, category=self.args.category)
        self.setup_properties_dict(train_dataset, properties_dict)
        return train_dataset, valid_dataset
        
    def build_predict_datasets(self, properties, properties_dict):
        dataset = read_fasta(self.args.data_path)
        pred_index_data = self.read_csv(self.args.predict_index_path)
        pred_dataset = PairwiseClassificationDataset(dataset, self.vocab, index_data=pred_index_data, category=self.args.category)
        self.predict_datasets = [pred_dataset]

    def build_test_datasets(self, properties, properties_dict, *args, **argv):
        dataset = read_fasta(self.args.data_path)
        test_index_data = self.read_csv(self.args.test_index_path)
        test_dataset = PairwiseClassificationDataset(dataset, self.vocab, index_data=test_index_data, category=self.args.category)
        self.test_datasets = [test_dataset]
    
    def load_properties_from_config(self, model_config, properties_dict):
        if self.args.category:
            properties_dict["label"] = getattr(model_config, "label_dict")
        return properties_dict

    # def set_collate_func(self, properties_dict, model_config, *args, **kwargs):
    #     # for label in self.args.labels:
    #     #     # setattr(self, "%s_vocab" % label, getattr(model_config, "%s_vocab" % label))
    #     #     # setattr(self, "%s_vocab" % label, getattr(model_config, "%s_dict" % label))
    #     #     properties_dict[label] = getattr(model_config, "%s_dict" % label)
    #     return super().set_collate_func(properties_dict, model_config, *args, **kwargs)

    def setup_properties_dict(self, train_dataset, properties_dict):
        if self.args.category or self.args.numerical:
            counter = Counter([x["label"] for x in train_dataset]).most_common()
            print(counter)
            vocab = list(set([x["label"] for x in train_dataset]))
            vocab.sort()
            label2index = {x: idx for idx, x in enumerate(vocab)}
            if not (len(vocab) == 1 and vocab[0] is None):
                properties_dict["label"] = label2index
                setattr(self.args, "label_vocab", vocab)
                setattr(self.args, "label_dict", label2index)
        setattr(self.args, "labels", ["label"])

@register_dm("hi_regression_aln")
class PairwiseRegressionAlnDataModule(PairwiseRegressionDataModule):
    def __init__(self, args, vocab=None):
        super().__init__(args, vocab)

    # @classmethod
    # def add_argparse_args(cls, parent_parser):
    #     parent_parser = super(PairwiseRegressionAlnDataModule, cls).add_argparse_args(parent_parser)
    #     # parent_parser.add_argument('--use_virus_msa', type=str2bool, default="false")
    #     # parent_parser.add_argument('--use_vaccine_msa', type=str2bool, default="false")
    #     return parent_parser

    def build_training_set(self, properties, properties_dict):
        train_data = self.read_csv(self.args.train_index_path)
        
        if self.args.train_loss_weight_path is not None:
            train_loss_weights = pd.read_csv(self.args.train_loss_weight_path)["loss_weight"]
        else:
            train_loss_weights = None
        
        # print(PairwiseAlnClassificationDataset)
        
        train_dataset = PairwiseAlnClassificationDataset(train_data, self.vocab, category=self.args.category, \
            prepend_special_token_for_seq1=self.args.prepend_special_token_for_virus,
            prepend_special_token_for_seq2=self.args.prepend_special_token_for_vaccine,
            numerical_interval=self.args.numerical_interval, numerical=self.args.numerical,
            loss_weights=train_loss_weights
            )
            
        if self.args.valid_loss_weight_path is not None:
            valid_loss_weights = pd.read_csv(self.args.valid_loss_weight_path)["loss_weight"]
        else:
            valid_loss_weights = None
        
        valid_data = self.read_csv(self.args.valid_index_path)
        valid_dataset = PairwiseAlnClassificationDataset(valid_data, self.vocab, category=self.args.category, \
            prepend_special_token_for_seq1=self.args.prepend_special_token_for_virus,
            prepend_special_token_for_seq2=self.args.prepend_special_token_for_vaccine,
            numerical_interval=self.args.numerical_interval, numerical=self.args.numerical,
            loss_weights=valid_loss_weights
            )
        self.setup_properties_dict(train_dataset, properties_dict)
        return train_dataset, valid_dataset
    
    # def build_predicting_set(self, pred_data_path, properties, properties_dict):
    #     meta_data = read_meta(self.args.meta_data_path)
    #     dataset = MultiTaskClassificationDataset(read_fasta(pred_data_path), self.vocab, meta_data=meta_data, classification_tasks=self.args.labels, binary=self.args.binary, predict=True)
    #     return dataset

    def build_test_datasets(self, properties, properties_dict, *args, **argv):
        test_data = self.read_csv(self.args.test_index_path)
        if self.args.test_loss_weight_path is not None:
            test_loss_weights = pd.read_csv(self.args.test_loss_weight_path)["loss_weight"]
        else:
            test_loss_weights = None
        test_dataset = PairwiseAlnClassificationDataset(test_data, self.vocab, category=self.args.category,
            prepend_special_token_for_seq1=self.args.prepend_special_token_for_virus,
            prepend_special_token_for_seq2=self.args.prepend_special_token_for_vaccine,
            numerical_interval=self.args.numerical_interval, 
            numerical=self.args.numerical, 
            loss_weights=test_loss_weights)
        self.test_datasets = [test_dataset]
    
    def build_predict_datasets(self, properties, properties_dict, *args, **argv):
        # print(self.args.predict_index_path)
        pred_data = self.read_csv(self.args.predict_index_path)
        pred_dataset = PairwiseAlnClassificationDataset(pred_data, self.vocab, category=self.args.category,
            prepend_special_token_for_seq1=self.args.prepend_special_token_for_virus,
            prepend_special_token_for_seq2=self.args.prepend_special_token_for_vaccine,
            numerical_interval=self.args.numerical_interval, numerical=self.args.numerical)
        self.pred_datasets = [pred_dataset]
  