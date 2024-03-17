import torch

def get_time_simple(desc):
    return float(desc.split()[-1])

def get_time_kw(desc, key="day"):
    desc_dict = {x.split("=")[0]: x.split("=")[1] for x in desc.split("|")}
    return float(desc_dict[key])

def get_value_from_desc(desc, key="day"):
    if len(desc) == 0:
        return None
    desc_dict = {x.split("=")[0]: x.split("=")[1] for x in desc.split("|")}
    try:
        return float(desc_dict[key])
    except Exception as e:
        if key in desc_dict:
            return desc_dict[key]
        else:
            return None

class TemporalFastaDataset(torch.utils.data.Dataset):
    def __init__(self, src_dataset, vocab, get_time_method="simple", properties=['day'], other_attributes=None) -> None:
        super().__init__()
        # Monolingual
        self.src_dataset = src_dataset
        self.vocab = vocab
        self.get_time_method = get_time_method
        self.other_attributes = other_attributes

        if get_time_method == "simple":
            self.get_time_func = get_time_simple
        elif get_time_method == "kw":
            self.get_time_func = get_value_from_desc
            self.properties = properties
        else:
            raise ValueError("Please set the right get_time_method.")

    def padding_all(self, pad_idx):
        max_len = max([x[1].size(0) for x in self.src_dataset])
        new_src_dataset = []
        new_tgt_dataset = []
        for data in self.src_dataset:
            tokens = data[1].new_zeros((max_len, ))
            tokens.fill_(pad_idx)
            tokens[:len(data[1])] = data[1]
            new_src_dataset.append((data[0], tokens, data[-1]))
        
        for data in self.tgt_dataset:
            tokens = data[1].new_zeros((max_len, ))
            tokens.fill_(pad_idx)
            tokens[:len(data[1])] = data[1]
            new_tgt_dataset.append((data[0], tokens, data[-1]))

        self.tgt_dataset = new_tgt_dataset
        self.src_dataset = new_src_dataset

        # self.processed_dataset = []
        # for index in range(len(self.src_dataset)):
        #     if self.get_time_method == "simple":
        #         ret = {
        #             "index": index,
        #             "src_id": self.src_dataset[index][0],
        #             "src_seq": self.src_dataset[index][1],
        #             "src_time": self.get_time_func(self.src_dataset[index][-1].split()[1]), # float(src[-1].split()[-1])
        #         }
        #     else:
        #         ret = {
        #             "index": index,
        #             "src_id": self.src_dataset[index][0],
        #             "src_seq": self.src_dataset[index][1],
        #         }
        #         desc = " ".join(self.src_dataset[index][-1].split()[1:])
        #         ret["src_time"] = self.get_time_func(desc, key=self.properties[0]) # float(src[-1].split()[-1])
        #         for key in self.properties[1:]:
        #             ret[key] = self.get_time_func(desc, key=key)
        #         # return ret
        #     self.processed_dataset.append(ret)

    def __len__(self, ):
        return len(self.src_dataset)
        
    def __getitem__(self, index):
        # return self.processed_dataset[index]
        if self.get_time_method == "simple":
            return {
                "index": index,
                "src_id": self.src_dataset[index][0],
                "src_seq": self.src_dataset[index][1],
                "src_time": self.get_time_func(self.src_dataset[index][-1].split()[1]), # float(src[-1].split()[-1])
            }
        else:
            ret = {
                "index": index,
                "src_id": self.src_dataset[index][0],
                "src_seq": self.src_dataset[index][1],
            }
            desc = " ".join(self.src_dataset[index][-1].split()[1:])
            ret["src_time"] = self.get_time_func(desc, key=self.properties[0]) # float(src[-1].split()[-1])
            for key in self.properties[1:]:
                ret[key] = self.get_time_func(desc, key=key)
            if self.other_attributes is not None:
                for key in self.other_attributes:
                    ret[key] = self.other_attributes[key][self.src_dataset[index][0]]
            return ret

class PairwiseClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, src_dataset, vocab, index_data, category=True) -> None:
        super().__init__()
        self.src_dataset = src_dataset
        self.vocab = vocab
        self.index_data = index_data
        self.category = category
        self.src_id_to_records = {x[0]: x for x in src_dataset}

    def __len__(self, ):
        return len(self.index_data)

    def __getitem__(self, index):
        # return self.processed_dataset[index]
        id1, id2, value = self.index_data[index][0], self.index_data[index][1], self.index_data[index][-1]
        if not self.category:
            value = float(value)
        seq = self.vocab.concat(self.src_id_to_records[id2][1], self.src_id_to_records[id1][1])
        ret = {
            "index": index,
            "src_id1": self.src_id_to_records[id1][0],
            "src_id2": self.src_id_to_records[id2][0],
            "src_seq": seq,
            "label": value
        }
        return ret

class PairwiseAlnClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, src_dataset, vocab, category=True, loss_weights = None,
    prepend_special_token_for_seq1=None,
    prepend_special_token_for_seq2=None,
    numerical=False, numerical_interval=None 
    ) -> None:
        # loss_weights: a list with the same size of src_dataset

        super().__init__()
        self.src_dataset = src_dataset
        self.vocab = vocab
        self.category = category
        self.numerical_interval = numerical_interval
        self.numerical = numerical
        self.prepend_special_token_for_seq1 = prepend_special_token_for_seq1
        self.prepend_special_token_for_seq2 = prepend_special_token_for_seq2
        if loss_weights is not None:
            assert len(loss_weights) == len(src_dataset), "Loss weights list (size=%d) should have the same size as src_dataset (size=%d)" % (len(loss_weights), len(src_dataset))
            self.loss_weights = loss_weights
        else:
            self.loss_weights = None

    def __len__(self, ):
        return len(self.src_dataset)

    def __getitem__(self, index):
        id1, id2, seq1, seq2, value = self.src_dataset[index]
        if not self.category and not self.numerical: # continuous
            value = float(value)
        elif self.numerical:
            value = int(float(value) // self.numerical_interval) #  * self.numerical_interval

        if "#" in seq1:
            seqs1 = seq1.split("#")
        else:
            seqs1 = [seq1]
        
        if self.prepend_special_token_for_seq1 is not None:
            seqs1 = [self.prepend_special_token_for_seq1 + x for x in seqs1]
        
        if "#" in seq2:
            seqs2 = seq2.split("#")
        else:
            seqs2 = [seq2]
                
        if self.prepend_special_token_for_seq2 is not None:
            seqs2 = [self.prepend_special_token_for_seq2 + x for x in seqs2]

        ret = {
            "index": index,
            "src_id1": id1,
            "src_id2": id2,
            "src_seq": seqs1 + seqs2,
            "label": value,
            "seq_label": [0] * len(seqs1) + [1] * len(seqs2),
            "ref_seq_label": [1] + [0] * (len(seqs1) - 1) + [1] + [0] * (len(seqs2) - 1),
            "loss_weight": self.loss_weights[index] if self.loss_weights is not None else 1.0
        }

        return ret