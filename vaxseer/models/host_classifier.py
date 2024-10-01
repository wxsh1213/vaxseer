from models import register_model
import torch, esm
import torch.nn as nn
import transformers
from pytorch_lightning import LightningModule
from esm import pretrained
from collections import defaultdict
from utils.args import str2bool
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
from esm.model.msa_transformer import MSATransformer


def configure_optimizers(model):
    """Prepare optimizer and schedule (linear warmup and decay)"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.config.learning_rate, eps=model.config.adam_epsilon)
    num_training_steps, num_warmup_steps = model.compute_warmup(
        num_training_steps=-1,
        num_warmup_steps=0.1,
    )
    if model.config.scheduler == "none":
        return [optimizer], []
    elif model.config.scheduler == "linear":
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    elif model.config.scheduler == "cosine":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    return [optimizer], [scheduler]

@register_model("esm_classifier")
class ESMClassifier(LightningModule):
    def __init__(self, config, alphabet, **args) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.alphabet = alphabet
        self.pad_idx = alphabet.pad()
        self.config = config

        if args.get("model_name_or_path", None) is None:
            model_name_or_path = config.model_name_or_path
        else:
            model_name_or_path = args.get("model_name_or_path")
        
        model_args = torch.load(model_name_or_path)
        model_args.layers = getattr(config, "n_layers", model_args.layers)
        self.alphabet = esm.Alphabet.from_architecture(model_args.arch)
        self.model = MSATransformer(model_args, self.alphabet)
        
        self.criterion = torch.nn.CrossEntropyLoss(reduce=False) # reduce=False
        if config.freeze_before_layer >= 0:
            for i, layer in enumerate(self.model.layers):
                if i < config.freeze_before_layer:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.classifier_heads = self.build_prediction_heads(config=config)

        if getattr(self.config, "add_seq_label_embeddings", False): #  self.config.:
            self.seq_label_embeddings = nn.Embedding(2, self.model.args.embed_dim) # self.args.embed_dim
        if getattr(self.config, "add_ref_seq_label_embeddings", False):
            self.ref_seq_label_embeddings = nn.Embedding(2, self.model.args.embed_dim)

    def overwrite_generate_kwargs(self, args):
        setattr(self.config, "predict_index_path", args.predict_index_path) 
        setattr(self.config, "predict_numerical_output", args.predict_numerical_output) 
        

    def build_prediction_heads(self, config):
        classifier_heads = nn.ParameterDict()
        for task in config.labels:
            classifier_heads[task] = nn.Sequential(
                nn.Linear(self.model.args.embed_dim, self.model.args.embed_dim),
                nn.LayerNorm(self.model.args.embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.model.args.embed_dim, len(getattr(config, task + "_dict"))),
            )
        return classifier_heads

    @classmethod
    def add_argparse_args(cls, parent_parser):

        parent_parser.add_argument('--freeze_before_layer', type=int, default=-1)
        parent_parser.add_argument('--repr_layer', type=int, default=-1)
        parent_parser.add_argument('--n_layers', type=int, default=12)
        parent_parser.add_argument('--model_name_or_path', type=str, default="models/esm_msa1b_t12_100M_UR50S_args.pkl")
        parent_parser.add_argument('--load_from_checkpoint', type=str, default=None)
        parent_parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        parent_parser.add_argument('--warmup_steps', type=int, default=0)
        parent_parser.add_argument('--weight_decay', type=float, default=0.0)
        parent_parser.add_argument('--dropout', type=float, default=0.1)

        parent_parser.add_argument('--add_seq_label_embeddings', type=str2bool, default="false")
        parent_parser.add_argument('--add_ref_seq_label_embeddings', type=str2bool, default="false")
        parent_parser.add_argument('--predict_numerical_output', type=str, default="average", choices=["argmax", "average"])
        
        return parent_parser

    def get_score(self, **args):
        results = self.model(**args, repr_layers=[self.config.repr_layer] if self.config.repr_layer is not None else [], return_contacts=False)
        hidden_states = results["representations"][self.config.repr_layer][:, 0] # CLS head
        if "msa" in self.config.model_name_or_path: # [B, C, L, V], C is the number of MSA
            hidden_states = hidden_states.mean(1)

        scores = {}
        for key in self.classifier_heads:
            logit_ = self.classifier_heads[key](hidden_states) # [B, 2]
            logit_ = torch.log_softmax(logit_, dim=-1) # [B, 2]
            score = logit_[:, getattr(self.config, "%s_dict" % key)['1']]
            scores[key] = score
        
        return scores

    def forward(self, batch, batch_idx, mode="train"):
        extra_embeddings = None

        if getattr(self.config, "add_seq_label_embeddings", False):
            extra_embeddings = self.seq_label_embeddings(batch["seq_label"]) # [B, M, H]
        if getattr(self.config, "add_ref_seq_label_embeddings", False):
            ref_seq_label_embeddings = self.ref_seq_label_embeddings(batch["ref_seq_label"]) # [B, M, H]
            if extra_embeddings is not None:
                extra_embeddings += ref_seq_label_embeddings
            else:
                extra_embeddings = ref_seq_label_embeddings
        if extra_embeddings is not None:
            extra_embeddings = extra_embeddings.unsqueeze(-2)

        results = self.model(
            batch["input_ids"], repr_layers=[self.config.repr_layer] if self.config.repr_layer is not None else [], return_contacts=False,
            extra_input_embeddings=extra_embeddings)
        hidden_states = results["representations"][self.config.repr_layer][:, 0] # CLS head
        if "msa" in self.config.model_name_or_path: # [B, C, L, V], C is the number of MSA
            hidden_states = hidden_states.mean(1)
         
        if mode == "test":
            loss_dict = {}
            for key in self.classifier_heads:
                logit_ = self.classifier_heads[key](hidden_states).squeeze(-1)
                loss_dict[key] = logit_
            return loss_dict
        else:
            loss_dict = {}
            loss_all = []
            # print(self.classifier_heads)
            for key in self.classifier_heads:
                logit_ = self.classifier_heads[key](hidden_states).squeeze(-1)
                loss = self.criterion(logit_, batch[key])
                loss_dict[key] = torch.mean(batch["loss_weight"] * loss)
                loss_all.append(loss_dict[key])
                loss_dict[key + "_no_reweight"] = torch.mean(loss)

            loss_all = sum(loss_all)
            return loss_all, loss_dict
    
    def testing_forward(self, batch):
        logits = self.forward(batch, None, mode="test")
        
        prediction_dict = dict()
        for key in self.classifier_heads:
            logit_ = logits[key]
            prob = torch.softmax(logit_, dim=-1)
            prediction_dict[key] = (prob, batch[key])

        return prediction_dict
    
    def output_predicting_results(self, outputs, predict_dataset, output_path):
        results = []
        for i in range(len(outputs)):
            results.append(outputs)

        results_cols = defaultdict(list)
        for output in outputs:
            for key in output:
                results_cols[key].append(output[key])

        # print(self.config.predict_index_path)
        ori_df = pd.read_csv(self.config.predict_index_path)
        ori_df = ori_df.drop(['virus_seq', 'reference_seq'], axis=1)
        for key in results_cols:
            ori_df[key] = results_cols[key]
        ori_df.to_csv(output_path, index=False)
        return None

    def pred_forward(self, batch):
        logits = self.forward(batch, None, mode="test") # [key == properties, value == tensor, predicted values] 
        prediction_dict = [{} for _ in range(batch["input_ids"].size(0))]
        for key in self.classifier_heads:
            if getattr(self.config, "numerical", False):
                numerical_values = torch.tensor(getattr(self.config, "%s_vocab"%key)).to(batch["input_ids"].device)
            logit_ = logits[key]

            if self.config.category or getattr(self.config, "numerical", False):
                prob = torch.softmax(logit_, dim=-1)
                if self.config.category:
                    for i, p in enumerate(prob):
                        for j in range(p.size(0)):
                            prediction_dict[i]["%s_%s" % (key, getattr(self.config, "%s_vocab"%key)[j])] = p[j].item()
                elif self.config.numerical:
                    for i, p in enumerate(prob):
                        if self.config.predict_numerical_output == "argmax":
                            prediction = numerical_values[torch.argmax(p)]
                        elif self.config.predict_numerical_output == "average":
                            prediction = torch.sum(p * numerical_values)
                        
                        prediction_dict[i][key] = prediction.item()
                            
                        for j in range(p.size(0)):
                            prediction_dict[i]["%s_%g" % (key, getattr(self.config, "%s_vocab"%key)[j])] = p[j].item()
            else:
                for i, p in enumerate(logit_):
                    prediction_dict[i]["%s" % (key)] = p.item()
        
        return prediction_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch, batch_idx, mode="train")
        self.log("train_loss", loss, prog_bar=True)
        for key in loss_dict:
            self.log("train_%s" % key, loss_dict[key], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch, batch_idx, mode="train")
        self.log("val_loss", loss, prog_bar=True)
        for key in loss_dict:
            self.log("val_%s" % key, loss_dict[key], prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self.testing_forward(batch)
        return loss_dict
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self.pred_forward(batch=batch)
        return predictions

    def test_epoch_end(self, outputs):
        results_pred = defaultdict(list)
        results_gt = defaultdict(list)

        for res in outputs:
            for key in res:
                results_pred[key].append(res[key][0])
                results_gt[key].append(res[key][1])
        
        for key in results_pred:
            preds = torch.cat(results_pred[key]).cpu()
            labels = torch.cat(results_gt[key]).cpu()
            prauc = average_precision_score(labels, preds[:, 1])
            prauc_rev = average_precision_score(1 - labels, preds[:, 0])
            roc_auc = roc_auc_score(labels, preds[:, 1])
            roc_auc_rev = roc_auc_score(1 - labels, preds[:, 0])
            self.log_dict({"%s_prauc" % key: prauc, \
                "%s_rocauc" % key: roc_auc, \
                "%s_prauc_rev" % key: prauc_rev, \
                "%s_roc_auc_rev" % key: roc_auc_rev})

    

    def configure_optimizers(self):
        return configure_optimizers(self)
        
    @property
    def num_training_steps(self) -> int:
        return self.trainer.estimated_stepping_batches

    def compute_warmup(self, num_training_steps, num_warmup_steps):
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

@register_model("esm_regressor_bilstm")
class biLSTMRegressor(LightningModule):
    def __init__(self, config, alphabet) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.alphabet = alphabet
        self.pad_idx = alphabet.pad()
        self.config = config

        self.embeddings = nn.Embedding(len(self.alphabet), config.hidden_size)
        self.rnn = nn.LSTM(config.hidden_size * 2, config.hidden_size, config.n_layer)
        self.criterion = torch.nn.MSELoss()
        self.out = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.LayerNorm(self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_size, 1),
            )
        

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser.add_argument('--hidden_size', type=int, default=512)
        parent_parser.add_argument('--n_layer', type=int, default=4)
        parent_parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        parent_parser.add_argument('--dropout', type=float, default=0.1)
        parent_parser.add_argument('--warmup_steps', type=int, default=0)
        parent_parser.add_argument('--weight_decay', type=float, default=0.0)
        return parent_parser

    def forward(self, batch, batch_idx, mode="train"):
        input_ids = batch["input_ids"]
        input_embeds = self.embeddings(batch["input_ids"].view(-1, input_ids.size(-1)))
        input_embeds = input_embeds.view(-1, 2, input_embeds.size(-2), input_embeds.size(-1))
        input_embeds = torch.cat([input_embeds[:, 0], input_embeds[:, 1]], dim=-1)

        h0 = input_embeds.new_zeros(self.config.n_layer, input_embeds.size(1), self.config.hidden_size)
        c0 = input_embeds.new_zeros(self.config.n_layer, input_embeds.size(1), self.config.hidden_size)
        output, (hn, cn) = self.rnn(input_embeds, (h0, c0))
        score = self.out(torch.mean(output, dim=1)).squeeze(-1)
        
        if mode != "test":
            loss = self.criterion(score, batch["label"])
            return loss, {}
        else:
            return score
        
    
    def output_predicting_results(self, outputs, predict_dataset, output_path):
        results = []
        for i in range(len(outputs)):
            results.append(outputs)

        results_cols = defaultdict(list)
        for output in outputs:
            for key in output:
                results_cols[key].append(output[key])

        ori_df = pd.read_csv(self.config.predict_index_path)
        ori_df = ori_df.drop(['virus_seq', 'reference_seq'], axis=1)
        for key in results_cols:
            ori_df[key] = results_cols[key]
        ori_df.to_csv(output_path, index=False)
        return None

    def pred_forward(self, batch):
        results = self.model(batch["input_ids"], repr_layers=[self.config.repr_layer] if self.config.repr_layer is not None else [], return_contacts=False)
        hidden_states = results["representations"][self.config.repr_layer][:, 0] # CLS head
        if "msa" in self.config.model_name_or_path: # [B, C, L, V], C is the number of MSA
            hidden_states = hidden_states.mean(1)
        
        prediction_dict = [{} for _ in range(batch["input_ids"].size(0))]
        for key in self.classifier_heads:
            logit_ = self.classifier_heads[key](hidden_states).squeeze(-1)
            if self.config.category:
                prob = torch.softmax(logit_, dim=-1)
                for i, p in enumerate(prob):
                    for j in range(p.size(0)):
                        prediction_dict[i]["%s_%s" % (key, getattr(self.config, "%s_vocab"%key)[j])] = p[j].item()
            else:
                for i, p in enumerate(logit_):
                    prediction_dict[i]["%s" % (key)] = p.item()
        return prediction_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch, batch_idx, mode="train")
        self.log("train_loss", loss, prog_bar=True)
        for key in loss_dict:
            self.log("train_%s" % key, loss_dict[key], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch, batch_idx, mode="train")
        self.log("val_loss", loss, prog_bar=True)
        for key in loss_dict:
            self.log("val_%s" % key, loss_dict[key], prog_bar=True)
        return loss
    
    def testing_forward(self, batch):
        logit = self.forward(batch, None, mode="test")
        return (logit, batch["label"])

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self.testing_forward(batch)
        return loss_dict
        # return {label: for label in self.config.labels}
        return loss_dict[self.config.labels[0]]
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self.pred_forward(batch=batch)
        return predictions

    def test_epoch_end(self, outputs):
        results_pred = []
        results_gt = []
        for res in outputs:
            results_pred.append(res[0])
            results_gt.append(res[1])
        
        
        preds = torch.cat(results_pred).cpu().squeeze()
        labels = torch.cat(results_gt).cpu().squeeze()
        labels_binary = (labels >= 3).long()

        prauc = average_precision_score(labels_binary, preds)
        roc_auc = roc_auc_score(labels_binary, preds)
        self.log_dict({"prauc_thres=3": prauc, \
            "rocauc_thres=3": roc_auc, \
            "mse": ((labels - preds) ** 2).mean()})
    
    def configure_optimizers(self):
        return configure_optimizers(self)
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=-1,
            num_warmup_steps=0.1,
        )

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    @property
    def num_training_steps(self) -> int:
        return self.trainer.estimated_stepping_batches

    def compute_warmup(self, num_training_steps, num_warmup_steps):
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps

@register_model("esm_regressor_cnn")
class CNNRegressor(LightningModule):
    def __init__(self, config, alphabet) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.alphabet = alphabet
        self.pad_idx = alphabet.pad()
        self.config = config

        # self.embeddings = nn.Embedding(len(self.alphabet), config.hidden_size)
        self.conv1 = nn.Conv1d(len(self.alphabet) * 2, 64, 15, stride=1)
        # self.pooling1 = nn.MaxPool2d(2, stride=1)
        self.pooling1 = nn.AdaptiveMaxPool1d(250)
        self.conv2 = nn.Conv1d(64, 64, 15, stride=1)
        self.pooling2 = nn.AdaptiveMaxPool1d(100)
        # self.conv1 = nn.Conv1d(2, 33, 3, stride=2)
        
        self.criterion = torch.nn.MSELoss()
        self.out = nn.Sequential(
                nn.Linear(64 * 100, self.config.hidden_size),
                nn.LayerNorm(self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_size, 1),
            )
        
    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser.add_argument('--hidden_size', type=int, default=512)
        parent_parser.add_argument('--adam_epsilon', type=float, default=1e-8)
        parent_parser.add_argument('--dropout', type=float, default=0.1)
        parent_parser.add_argument('--warmup_steps', type=int, default=0)
        parent_parser.add_argument('--weight_decay', type=float, default=0.0)
        return parent_parser

    def overwrite_generate_kwargs(self, args):
        setattr(self.config, "predict_index_path", args.predict_index_path) 

    def get_score(self, **args):
        results = self.model(**args, repr_layers=[self.config.repr_layer] if self.config.repr_layer is not None else [], return_contacts=False)
        hidden_states = results["representations"][self.config.repr_layer][:, 0] # CLS head
        if "msa" in self.config.model_name_or_path: # [B, C, L, V], C is the number of MSA
            hidden_states = hidden_states.mean(1)

        scores = {}
        for key in self.classifier_heads:
            logit_ = self.classifier_heads[key](hidden_states) # [B, 2]
            logit_ = torch.log_softmax(logit_, dim=-1) # [B, 2]
            score = logit_[:, getattr(self.config, "%s_dict" % key)['1']]
            scores[key] = score
        
        return scores

    def testing_forward(self, batch):
        logit = self.forward(batch, None, mode="test")
        return (logit, batch["label"])

    def forward(self, batch, batch_idx, mode="train"):
        input_ids = batch["input_ids"]
        input_embeds = torch.nn.functional.one_hot(input_ids, num_classes=len(self.alphabet)).float()
        input_embeds = torch.cat([input_embeds[:, 0], input_embeds[:, 1]], dim=-1)
        input_embeds = input_embeds.transpose(-2, -1)
        out = self.conv1(input_embeds)
        out = self.pooling1(out)
        out = self.conv2(out)
        out = self.pooling2(out) # [B, 64, 100]
        out = out.view(out.size(0), -1)
        score = self.out(out).squeeze(-1)
        if mode != "test":
            loss = self.criterion(score, batch["label"])
            return loss, {}
        else:
            return score
        
    
    def output_predicting_results(self, outputs, predict_dataset, output_path):
        results = []
        for i in range(len(outputs)):
            results.append(outputs)

        results_cols = defaultdict(list)
        for output in outputs:
            for key in output:
                results_cols[key].append(output[key])

        ori_df = pd.read_csv(self.config.predict_index_path)
        ori_df = ori_df.drop(['virus_seq', 'reference_seq'], axis=1)
        for key in results_cols:
            ori_df[key] = results_cols[key]
        ori_df.to_csv(output_path, index=False)
        return None

    def pred_forward(self, batch):
        input_ids = batch["input_ids"]
        input_embeds = torch.nn.functional.one_hot(input_ids, num_classes=len(self.alphabet)).float()
        input_embeds = torch.cat([input_embeds[:, 0], input_embeds[:, 1]], dim=-1)
        input_embeds = input_embeds.transpose(-2, -1)
        out = self.conv1(input_embeds)
        out = self.pooling1(out)
        out = self.conv2(out)
        out = self.pooling2(out) # [B, 64, 100]
        out = out.view(out.size(0), -1)
        score = self.out(out).squeeze(-1)

        prediction_dict = [{} for _ in range(batch["input_ids"].size(0))]
        for i, p in enumerate(score):
            prediction_dict[i]["label"] = p.item()
        return prediction_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch, batch_idx, mode="train")
        self.log("train_loss", loss, prog_bar=True)
        for key in loss_dict:
            self.log("train_%s" % key, loss_dict[key], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch, batch_idx, mode="train")
        self.log("val_loss", loss, prog_bar=True)
        for key in loss_dict:
            self.log("val_%s" % key, loss_dict[key], prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss_dict = self.testing_forward(batch)
        return loss_dict
        # return {label: for label in self.config.labels}
        return loss_dict[self.config.labels[0]]
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self.pred_forward(batch=batch)
        return predictions

    def test_epoch_end(self, outputs):
        results_pred = []
        results_gt = []
        for res in outputs:
            results_pred.append(res[0])
            results_gt.append(res[1])
        
        
        preds = torch.cat(results_pred).cpu().squeeze()
        labels = torch.cat(results_gt).cpu().squeeze()
        labels_binary = (labels >= 3).long()

        self.all_outputs = preds

        prauc = average_precision_score(labels_binary, preds)
        roc_auc = roc_auc_score(labels_binary, preds)
        self.log_dict({"prauc_thres=3": prauc, \
            "rocauc_thres=3": roc_auc, \
            "mse": ((labels - preds) ** 2).mean(), "rmse": torch.sqrt(((labels - preds) ** 2).mean())
            })
    
    def output_testing_results(self, outputs, test_datasets):
        assert len(test_datasets) == 1
        test_dataset = test_datasets[0]
        output_set = []
        for i, item in enumerate(test_dataset):
            new_item = {}
            new_item["src_id1"] = item["src_id1"]
            new_item["src_id2"] = item["src_id2"]
            new_item["pred_label"] = outputs[i].item()
            new_item["label"] = item["label"]
            output_set.append(new_item)
        return output_set
    
    def configure_optimizers(self):
        return configure_optimizers(self)
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate, eps=self.config.adam_epsilon)
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=-1,
            num_warmup_steps=0.1,
        )
        if self.config.scheduler == "none":
            return {
                "optimizer": optimizer,
            }
        elif self.config.scheduler == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )
        elif self.config.scheduler == "cosine":
            scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
    
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    @property
    def num_training_steps(self) -> int:
        return self.trainer.estimated_stepping_batches

    def compute_warmup(self, num_training_steps, num_warmup_steps):
        if num_training_steps < 0:
            # less than 0 specifies to infer number of training steps
            num_training_steps = self.num_training_steps
        if isinstance(num_warmup_steps, float):
            # Convert float values to percentage of training steps to use as warmup
            num_warmup_steps *= num_training_steps
        return num_training_steps, num_warmup_steps


@register_model("esm_regressor")
class ESMRegressor(ESMClassifier):
    def __init__(self, config, alphabet, **args) -> None:
        super().__init__(config, alphabet, **args)
        self.criterion = torch.nn.MSELoss(reduce=False)

    def testing_forward(self, batch):
        logits = self.forward(batch, None, mode="test")
        prediction_dict = dict()
        for key in self.classifier_heads:
            prediction_dict[key] = (logits[key], batch[key])
        return prediction_dict

    def build_prediction_heads(self, config):
        classifier_heads = nn.ParameterDict()
        for task in config.labels:
            classifier_heads[task] = nn.Sequential(
                nn.Linear(self.model.args.embed_dim, self.model.args.embed_dim),
                nn.LayerNorm(self.model.args.embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.model.args.embed_dim, 1),
            )
        return classifier_heads
    
    def output_testing_results(self, outputs, test_datasets):
        assert len(test_datasets) == 1
        test_dataset = test_datasets[0]
        output_set = []
        for i, item in enumerate(test_dataset):
            new_item = {}
            new_item["src_id1"] = item["src_id1"]
            new_item["src_id2"] = item["src_id2"]
            for key in outputs:
                new_item["pred_" + key] = outputs[key][i].item()
                new_item[key] = item[key]
            output_set.append(new_item)
        return output_set
        
    def test_epoch_end(self, outputs):
        results_pred = defaultdict(list)
        results_gt = defaultdict(list)
        for res in outputs:
            for key in res:
                results_pred[key].append(res[key][0])
                results_gt[key].append(res[key][1])
        
        self.all_outputs = {}
        for key in results_pred:
            preds = torch.cat(results_pred[key]).cpu().squeeze()
            labels = torch.cat(results_gt[key]).cpu().squeeze()
            labels_binary = (labels >= 3).long()
            
            prauc = average_precision_score(labels_binary, preds)
            roc_auc = roc_auc_score(labels_binary, preds)
            self.log_dict({"%s_prauc_thres=3" % key: prauc, \
                "%s_rocauc_thres=3" % key: roc_auc, \
                "%s_mse" % key: ((labels - preds) ** 2).mean(), \
                "%s_rmse" % key: torch.sqrt(((labels - preds) ** 2).mean())
                })
            
            self.all_outputs[key] = preds