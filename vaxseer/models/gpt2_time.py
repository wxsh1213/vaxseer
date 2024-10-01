import torch
import pytorch_lightning as pl
from lightning_transformers.task.nlp.language_modeling import (
    LanguageModelingTransformer,
)
import transformers
import torch.nn as nn
from data.utils import discretize_time
from torch.nn import CrossEntropyLoss
from models import register_model
import math, logging
from typing import IO, Any, Callable, Dict, Optional, Tuple, Type, Union
from utils.args import str2bool
from transformers import AutoConfig, PreTrainedTokenizerBase

class GPT2TimeModel(transformers.GPT2LMHeadModel):
    def __init__(self, config) -> None:
        super().__init__(config)

        # if not config.zero_offset:
        self.build_offset_layer(config)
        self.config = config
    
    def build_offset_layer(self, config):
        if getattr(self.config, "transformer_offset", False):
            self.offset_layer = transformers.GPT2LMHeadModel(config) 
        else:
            self.offset_layer = nn.Linear(config.hidden_size, config.vocab_size) # Just a linear layer

    @classmethod
    def from_config(cls, config):
        model = cls(config)
        return model
    
    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method.
        """
        return {"input_ids": input_ids, "input_time": kwargs["input_time"], "logits_processor": kwargs.get("logits_processor", None)} # "input_time": kwargs["input_time"], 

    def get_offset(self, outputs=None, **argv):
        if self.config.transformer_offset:
            outputs = self.offset_layer.forward(input_ids = argv.get("input_ids"), labels = argv.get("labels"), \
                attention_mask = argv.get("attention_mask"), output_hidden_states=True)
            offset = outputs.logits
        else:
            hidden_states = outputs.hidden_states[-1]
            offset = self.offset_layer(hidden_states)
        
        return offset
    
    def forward(self, input_time, return_hidden_states=False, **argv):
        time = discretize_time(
            input_time, 
            one_step=False, 
            normalize_time_a=self.config.normalize_time_a, 
            normalize_time_b=self.config.normalize_time_b,
            discrete=False)
        beam_size = argv.get("input_ids").size(0) // input_time.size(0)
        time = time.unsqueeze(1).repeat(1, beam_size).view(-1)
        outputs = super().forward(input_ids = argv.get("input_ids"), labels = argv.get("labels"), \
            attention_mask = argv.get("attention_mask"), output_hidden_states=True)
        
        rate = outputs.logits
        logits = rate * time.unsqueeze(-1).unsqueeze(-1)
        offset = self.get_offset(outputs, **argv)
        logits = logits + offset
        outputs.logits = logits

        if return_hidden_states:
            outputs.hidden_states = outputs.hidden_states

        return outputs

@register_model("gpt2_time_new")
class GPT2TimeNew(LanguageModelingTransformer):
    def __init__(self, config, alphabet, **kwargs) -> None:
        self.config = config
        super().__init__(
            pretrained_model_name_or_path=config.model_name_or_path, # GPT-2
            load_weights=config.load_weights,  # False
            vocab_size=len(alphabet) if kwargs.get("vocab_size") is None else kwargs.get("vocab_size"),  # TODO: build the alphabet first!!!!!!!!!!
            max_position_embeddings=config.max_position_embeddings, # 1024 by default, but please set larger.
            num_hidden_layers=config.num_hidden_layers, # 12
            hidden_size=config.hidden_size # 768
            )
        
        self.alphabet = alphabet
        self.pad_idx = alphabet.pad()
        if getattr(config, "load_from_pretrain_checkpoint", None):
            self.load_pretrained_model(config.load_from_pretrain_checkpoint)
    

    def initialize_model(self, pretrained_model_name_or_path: str):
        """create and initialize the model to use with this task,
        Feel free to overwrite this method if you are initializing the model in a different way
        """
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path="gpt2", **self.model_data_kwargs
        )
        setattr(config, "normalize_time_a", self.config.normalize_time_a)
        setattr(config, "normalize_time_b", self.config.normalize_time_b)
        setattr(config, "transformer_offset", self.config.transformer_offset)
        self.model = GPT2TimeModel.from_config(config)

    def load_pretrained_model(self, path):
        pretrained_model_state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for state in pretrained_model_state_dict:
            if state in self.state_dict():
                if self.state_dict()[state].size() != pretrained_model_state_dict[state].size():
                    logging.warning("The parameter %s of pretrained model (%s) doesn't fit the current model %s." % (state, str(pretrained_model_state_dict[state].size()), str(self.state_dict()[state].size())))
                else:
                    self.state_dict()[state].copy_(pretrained_model_state_dict[state])

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
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

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, IO],
        map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
        hparams_file: Optional[str] = None,
        strict: bool = True,
        hf_pipeline_kwargs: Optional[Dict] = None,
        # config = None,
        args = None,
        **kwargs
    ):

        model = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict)
        # model.resume_from_checkpoint = checkpoint_path
        model.config.resume_from_checkpoint = checkpoint_path
        model.config.pred_data_paths = getattr(args, "pred_data_paths", "")
        if args is not None:
            model.config.test_data_paths = args.test_data_paths
        for key in kwargs:
            logging.info("Overwrite model hyperparameter %s:" % key + ", from " + str(getattr(model, key, None)) + " to " + str(kwargs[key]))
            setattr(model, key, kwargs[key])
        return model

    @classmethod
    def add_argparse_args(cls, parent_parser):
        # For testing
        parent_parser.add_argument('--load_weights', action='store_true')
        parent_parser.add_argument('--num_hidden_layers', type=int, default=12)
        parent_parser.add_argument('--tau', type=float, default=1.0, help="Devide t by tau.")
        parent_parser.add_argument('--hidden_size', type=int, default=768)
        parent_parser.add_argument('--model_name_or_path', type=str, default="gpt2")
        parent_parser.add_argument('--load_from_pretrain_checkpoint', type=str, default=None)
        # For time embeddings
        parent_parser.add_argument('--normalize_time_a', type=int, default=1,  help="t = (t-b)/a")
        parent_parser.add_argument('--normalize_time_b', type=int, default=0, help="t = (t-b)/a")
        parent_parser.add_argument('--add_location', action='store_true', help="Add the location information.")
        parent_parser.add_argument('--add_lineage', action='store_true', help="Add the lineage information.")
        # Settings for the off-set layer:
        parent_parser.add_argument('--weight_loss_by_count', type=str2bool, default="false", help="Weight loss of each sample by their counting not frequency")
        parent_parser.add_argument('--no_normalization_in_batch', action='store_true', help="Don't normalize the loss weight within the batch!!")
        parent_parser.add_argument('--zero_offset', action='store_true', help="Set the sequences distribution at offset as 0")
        parent_parser.add_argument('--offset_share_layer', type=int, default=-1, help="Use the hidden state at layer i to output the offset.")
        parent_parser.add_argument('--transformer_offset', action='store_true', help="Use another transformer NN to predict the offset.")
        parent_parser.add_argument('--second_order_rate', action='store_true', help="Add the second order rate in modeling.")
        parent_parser.add_argument('--transformer_second_order_rate', action='store_true', help="Add the second order rate in modeling.")
        parent_parser.add_argument('--output_token_losses', type=str2bool, default="false")

        parent_parser.add_argument('--do_sample', type=str2bool, default="false")
        parent_parser.add_argument('--temperature', type=float, default=1.0)
        parent_parser.add_argument('--num_beams', type=int, default=1)
        parent_parser.add_argument('--num_return_sequences', type=int, default=1)

        parent_parser.add_argument('--zero_time', action='store_true', help="Set the time as zero.")
        parent_parser.add_argument('--set_time', type=float, default=None)
        
        parent_parser.add_argument('--ensemble', type=str2bool, default="false")
        parent_parser.add_argument('--average_over_time', type=str2bool, default="false")

        parent_parser.add_argument('--freeze_params_before_layer', type=int, default=0)
        parent_parser.add_argument('--weight_loss_by_time', type=str2bool, default="false")
        return parent_parser
        
    def nll_loss(self, lm_logits, labels, loss_weight=None, reduce=True):
        labels = labels.masked_fill(torch.eq(labels, self.alphabet.pad()), -100)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduce=False)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        if reduce:
            loss = loss.sum(dim=-1) / (shift_labels != -100).sum(dim=-1) # [B]
            if loss_weight is not None:
                if not self.config.no_normalization_in_batch:
                    loss_weight = loss_weight / loss_weight.sum()
                loss = torch.sum(loss * loss_weight)
            else:
                loss = loss.mean()
        
        return loss

    def get_offset(self, batch, outputs=None):
        if getattr(self.config, "transformer_offset", False):
            offset = self.offset_layer.model(input_ids = batch["input_ids"], labels = batch["labels"], attention_mask = batch["attention_mask"]).logits
        else:
            assert outputs is not None
            hidden_states = outputs.hidden_states[getattr(self.config, "offset_share_layer", -1)]
            offset = self.offset_layer(hidden_states)
        return offset

    def get_unnorm_nll(self, rate_logits, labels, reduce=True):
        loss = - nn.NLLLoss(reduce=False)(rate_logits.view(-1, rate_logits.size(-1)), labels.view(-1))
        loss = loss.view(labels.size())
        if reduce:
            return loss.sum(-1)
        else:
            return loss

    def core(self, batch):
        inputs_embeds = self.model.transformer.wte(batch["input_ids"])
        if self.config.add_location:
            loc_embeds = self.location_embeddings(batch["location"])
            inputs_embeds = inputs_embeds + loc_embeds.unsqueeze(1)
        if getattr(self.config, "add_lineage", False):
            lineage_embeds = self.lineage_embeddings(batch["lineage"]) 
            inputs_embeds = inputs_embeds + lineage_embeds.unsqueeze(1)
        outputs = self.model(inputs_embeds = inputs_embeds, labels = batch["labels"], attention_mask = batch["attention_mask"], output_hidden_states=True)
        return outputs
    
    def get_rate(self, outputs):
        return outputs.logits

    def testing_forward(self, batch, batch_idx, return_rate=False, return_offset=False):
        loss_weight = batch.get('freq', None)
        max_time, min_time = self.max_testing_time, self.min_testing_time
        input_times = torch.arange(min_time, max_time + 1).to(batch["input_ids"].device)
        
        time = discretize_time(
            input_times, 
            one_step=False, 
            normalize_time_a=self.config.normalize_time_a, 
            normalize_time_b=self.config.normalize_time_b,
            discrete=False)

        outputs = self.core(batch)
        rate = self.get_rate(outputs).unsqueeze(0) # [1, B, L, V], time: [T, 1, 1, 1]
        logits = rate * time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / getattr(self.config, "tau", 1.0) # [B, L, V]
        if not self.config.zero_offset:
            offset = self.get_offset(batch, outputs).unsqueeze(0)
            logits = logits + offset
        else:
            logits = logits
        # logits: [T, B, L, V]
        labels = batch["labels"]# [T, B, L]
        labels = labels.masked_fill(torch.eq(labels, self.alphabet.pad()), -100)
       
        repeat_labels = labels.unsqueeze(0).repeat(logits.size(0), 1, 1)  
        loss = self.nll_loss(logits.view(-1, logits.size(2), logits.size(3)), \
            repeat_labels.view(-1, repeat_labels.size(2)), loss_weight=loss_weight, reduce=False)
        loss = loss.view(logits.size(0), -1) # [T, B]
        
        loss_dict = {}
        if return_rate:
            loss_dict["rate"] = self.get_unnorm_nll(rate.squeeze(0), labels)
        if return_offset and not self.config.zero_offset:
            loss_dict["offset"] = self.get_unnorm_nll(offset, labels)

        return loss, loss_dict

    def forward(self, batch, batch_idx, reduce=True, return_rate=False, return_offset=False, mode="train"):
        if getattr(self.config, "zero_time", False):
            batch["input_time"].fill_(0.)

        if getattr(self.config, "set_time", None) is not None:
            batch["input_time"].fill_(self.config.set_time) # set time bin as a constant.

        logits = self.model(**batch).logits / self.config.temperature
        if self.config.weight_loss_by_count and batch.get('freq', None) is not None and batch.get('bin_size', None) is not None:
            loss_weight = batch.get('freq', None) * batch.get('bin_size', None)
        elif not self.config.weight_loss_by_count and batch.get('freq', None) is not None:
            loss_weight = batch.get('freq', None)
        else:
            loss_weight = 1.0
        
        labels = batch["labels"]
        loss = self.nll_loss(logits, labels, loss_weight=loss_weight, reduce=reduce)

        return loss, {}

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        for key in loss_dict:
            self.log("train_%s" % key, loss_dict[key], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        for key in loss_dict:
            self.log("val_%s" % key, loss_dict[key], prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, loss_dict = self.forward(batch, batch_idx, reduce=False, mode="test")
        token_num = torch.sum(
            (batch["labels"][..., 1:].contiguous() != self.alphabet.pad()) * 
            (batch["labels"][..., 1:].contiguous() != self.alphabet.eos()) * 
            (batch["labels"][..., 1:].contiguous() != self.alphabet.bos()), dim=-1)

        if "freq" in batch and "bin_size" in batch:
            weight = batch["freq"] * batch["bin_size"]
        else:
            weight = token_num.new_zeros(token_num.size(0)) + 1.0
        self.log("test_loss", loss.mean(), prog_bar=True)
        return loss, token_num, weight

    def overwrite_generate_kwargs(self, new_config):
        setattr(self.config, "do_sample", new_config.do_sample)
        setattr(self.config, "num_beams", new_config.num_beams)
        setattr(self.config, "temperature", new_config.temperature)
        setattr(self.config, "num_return_sequences", new_config.num_return_sequences)
        setattr(self.config, "output_token_losses", new_config.output_token_losses)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        generate_kwargs = {}
        generate_kwargs["temperature"] = getattr(self.config, "temperature", 1.0) # TODO: how to add this in testing?
        generate_kwargs["do_sample"] = getattr(self.config, "do_sample", True)
        generate_kwargs["num_beams"] = getattr(self.config, "num_beams", 1.0)
        setattr(self.model.config, "num_beams", generate_kwargs["num_beams"])

        generate_kwargs["num_return_sequences"] = max(getattr(self.config, "num_return_sequences", 1.0), generate_kwargs["num_beams"])

        if getattr(self.config, "generate_max_length", None) is None:
            generate_kwargs["max_length"] = self.config.max_position_embeddings
        else:
            generate_kwargs["max_length"] = getattr(self.config, "generate_max_length", None)
        generate_kwargs["pad_token_id"] = self.alphabet.pad()
        generate_kwargs["eos_token_id"] = self.alphabet.eos()
        generate_kwargs["bos_token_id"] = self.alphabet.bos()
        
        if batch["input_ids"][0, -1].item() == self.alphabet.eos():
            batch["input_ids"] = batch["input_ids"][:, :-1]

        output_ids = self.model.generate(**batch, **generate_kwargs)
        input_time = batch["input_time"].unsqueeze(1).repeat(1, generate_kwargs["num_return_sequences"]).view(-1)
        outputs = [{"prediction": self.alphabet.string(x), "src_time": input_time[i].item()} for i, x in enumerate(output_ids)]
        return outputs
        
    def test_epoch_end(self, outputs):
        losses, token_nums, weights = [], [], []
        if len(self.config.test_data_paths) == 1:
            outputs = [outputs]

        for dataloader_outputs in outputs:
            for output in dataloader_outputs:
                # outpu[0]: [B, L]
                losses.append(output[0].sum(-1)) # [B]
                token_nums.append(output[1])
                weights.append(output[2])
        losses = torch.cat(losses)
        token_nums = torch.cat(token_nums)
        weights = torch.cat(weights)
        ppl = torch.exp(torch.sum(losses * weights) / torch.sum(token_nums * weights))
        nll = torch.sum(weights * losses) / torch.sum(weights)
        self.log_dict({"perplexity": ppl, "nll": nll, "coverage": torch.exp(-losses).sum()})

        if self.config.output_token_losses:
            self.all_outputs = []
            for dataloader_outputs in outputs:
                for output in dataloader_outputs:
                    self.all_outputs.extend([x for x in output[0]])
        else:
            self.all_outputs = []
            for loss, tok_num in zip(losses, token_nums):
                self.all_outputs.append({"prediction": loss.item(), "token_num": tok_num.item()})

        return ppl
    
    def output_testing_results(self, outputs, predict_dataset):
        
        predict_dataset = [item for sublist in predict_dataset for item in sublist]
        assert len(outputs) == len(predict_dataset)
        results = []
        for index, output_loss in enumerate(outputs):
            # src_id,freq,src_time,prediction,rate,offset
            if self.config.output_token_losses:
                output_dict = {"prediction": " ".join([str(x.item()) for x in output_loss])}
            else:
                output_dict = output_loss
            output_dict["src_id"] = predict_dataset[index]["src_id"]
            output_dict["src_time"] = predict_dataset[index]["src_time"]
            output_dict["freq"] = predict_dataset[index]["freq"]
            results.append(output_dict)
        return results

    def output_predicting_results(self, outputs, predict_dataset, *args, **kwargs):
        results = []
        for i, output_dict in enumerate(outputs):
            # src_id,freq,src_time,prediction,rate,offset
            output_dict["prediction"] = output_dict["prediction"]
            output_dict["src_time"] = output_dict["src_time"]
            results.append(output_dict)
        return results

        results = []
        for output_dict in outputs:
            index = output_dict["index"]
            # src_id,freq,src_time,prediction,rate,offset
            output_dict["src_id"] = predict_dataset[index]["src_id"]
            output_dict["freq"] = predict_dataset[index]["freq"]
            results.append(output_dict)
        return results