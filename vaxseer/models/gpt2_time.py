import torch
import pytorch_lightning as pl
from lightning_transformers.task.nlp.language_modeling import (
    # LanguageModelingDataModule,
    LanguageModelingTransformer,
)
import transformers
# from pytorch_lightning import LightningModule
# from pytorch_lightning.utilities import rank_zero_warn
import torch.nn as nn
# from torch.distributions.gamma import Gamma
from data.utils import discretize_time
from torch.nn import CrossEntropyLoss
from models import register_model
import math, logging
# from esm import pretrained
from typing import IO, Any, Callable, Dict, Optional, Tuple, Type, Union
from utils.args import str2bool
# from transformers import GPT2LMHeadModel
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
        # print(input_ids)
        # print(self.config.prepend_property)
        # print(self.config.vocab_size)
        
        # if getattr(self.config, "prepend_property", False) and input_ids[0, 0] < self.config.vocab_size:
        #     offset = self.config.vocab_size
        #     prepend_ids = []
        #     for prop in self.config.data_properties:
        #         prop_tok = kwargs[prop] + offset
        #         offset += len(getattr(self.config, "%s_dict" % prop))
        #         # print(prop_tok.unsqueeze(-1).size(), input_ids.size())
        #         prepend_ids.append(prop_tok)
        #         # input_ids = torch.cat([prop_tok.unsqueeze(-1), input_ids], dim=-1)
        #         # print(input_ids)
        #     prepend_ids = torch.stack(prepend_ids, dim=1)
        #     # print(prepend_ids.size())
        #     input_ids = torch.cat([prepend_ids, input_ids], dim=-1)
        #     print(input_ids.size())
        #     print(input_ids)
        #     exit()

        # print(kwargs["attention_mask"])
        return {"input_ids": input_ids, "input_time": kwargs["input_time"], "logits_processor": kwargs.get("logits_processor", None)} # "input_time": kwargs["input_time"], 

    def get_offset(self, outputs=None, **argv):
        # print(argv)
        # print(torch.sum(argv["input_ids"][0] != argv["input_ids"][3]))
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
        # print(time)
        beam_size = argv.get("input_ids").size(0) // input_time.size(0)
        time = time.unsqueeze(1).repeat(1, beam_size).view(-1)
        # print(time)
        outputs = super().forward(input_ids = argv.get("input_ids"), labels = argv.get("labels"), \
            attention_mask = argv.get("attention_mask"), output_hidden_states=True)
        # print(outputs)
        # print(len(outputs))
        
        rate = outputs.logits
        # print(rate.size(), time.size())
        # print(argv.get("input_ids"))
        # exit()
        logits = rate * time.unsqueeze(-1).unsqueeze(-1)
        offset = self.get_offset(outputs, **argv)
        # print(logits.size(), offset.size())
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
        
        # if not config.zero_offset:
        #     self.build_offset_layer(config)
        # if getattr(config, "second_order_rate", False):
        #     self.second_order_rate_layer = self.build_second_order_rate(config)

        # if getattr(config, "add_location", False): # .add_location:
        #     self.location_embeddings = nn.Embedding(len(config.location_list), config.hidden_size)
        #     # self.embeddings_nn = nn.Linear(config.hidden_size*2, config.hidden_size)
        # if getattr(config, "add_lineage", False):
        #     self.lineage_embeddings = nn.Embedding(len(config.lineage_to_index), config.hidden_size)
        
        if getattr(config, "load_from_pretrain_checkpoint", None):
            self.load_pretrained_model(config.load_from_pretrain_checkpoint)
    

    def initialize_model(self, pretrained_model_name_or_path: str):
        """create and initialize the model to use with this task,
        Feel free to overwrite this method if you are initializing the model in a different way
        """
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path="gpt2", **self.model_data_kwargs
        )
        # print(config.vocab_size)
        setattr(config, "normalize_time_a", self.config.normalize_time_a)
        setattr(config, "normalize_time_b", self.config.normalize_time_b)
        setattr(config, "transformer_offset", self.config.transformer_offset)
        self.model = GPT2TimeModel.from_config(config)

    # def build_offset_layer(self, config):
    #     if getattr(self.config, "transformer_offset", False):
    #         self.offset_layer = LanguageModelingTransformer(
    #             pretrained_model_name_or_path=config.model_name_or_path,
    #             load_weights=config.load_weights,
    #             vocab_size=len(self.alphabet),
    #             max_position_embeddings=config.max_position_embeddings,
    #             num_hidden_layers=config.num_hidden_layers,
    #             hidden_size=config.hidden_size) 
    #     else:
    #         self.offset_layer = nn.Linear(config.hidden_size, len(self.alphabet)) # Just a linear layer

    def load_pretrained_model(self, path):
        pretrained_model_state_dict = torch.load(path, map_location="cpu")["state_dict"]
        for state in pretrained_model_state_dict:
            if state in self.state_dict():
                if self.state_dict()[state].size() != pretrained_model_state_dict[state].size():
                    logging.warning("The parameter %s of pretrained model (%s) doesn't fit the current model %s." % (state, str(pretrained_model_state_dict[state].size()), str(self.state_dict()[state].size())))
                else:
                    self.state_dict()[state].copy_(pretrained_model_state_dict[state])

    def configure_optimizers(self) -> Dict:
        # rank_zero_warn(
        #     "You haven't specified an optimizer or lr scheduler. "
        #     "Defaulting to AdamW with an lr of 1e-5 and linear warmup for 10% of steps. "
        #     "To change this, override ``configure_optimizers`` in  TransformerModule."
        # )
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)
        num_training_steps, num_warmup_steps = self.compute_warmup(
            num_training_steps=-1,
            num_warmup_steps=0.1,
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
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
        # if args.ensemble and len(checkpoint_path.split(",")) > 1:
        #     checkpoint_paths = checkpoint_path.split(",")
        #     # print(checkpoint_paths)
        #     model_list = []
        #     for path in checkpoint_paths:
        #         _model = super().load_from_checkpoint(path, map_location, hparams_file, strict)
        #         model_list.append(_model)
        #     # models = [super().load_from_checkpoint(path, map_location, hparams_file, strict) for path in checkpoint_paths]
        #     model = nn.ModuleList(model_list) 
        # else:
        
        # hparams_file=checkpoint_path.split("/checkpoints/")[0] + "/hparams.yaml"
        # print(hparams_file)
        # print(checkpoint_path)

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
        # parent_parser = super(myGPT2, cls).add_argparse_args()
        # For testing
        parent_parser.add_argument('--load_weights', action='store_true')
        parent_parser.add_argument('--num_hidden_layers', type=int, default=12)
        parent_parser.add_argument('--tau', type=float, default=1.0, help="Devide t by tau.")
        parent_parser.add_argument('--hidden_size', type=int, default=768)
        parent_parser.add_argument('--model_name_or_path', type=str, default="gpt2")
        parent_parser.add_argument('--load_from_pretrain_checkpoint', type=str, default=None)
        # parent_parser.add_argument('--max_position_embeddings', type=int, default=1280)
        # For time embeddings
        parent_parser.add_argument('--normalize_time_a', type=int, default=1,  help="t = (t-b)/a")
        parent_parser.add_argument('--normalize_time_b', type=int, default=0, help="t = (t-b)/a")
        # parent_parser.add_argument('--time_agnostic', action='store_true')
        parent_parser.add_argument('--add_location', action='store_true', help="Add the location information.")
        parent_parser.add_argument('--add_lineage', action='store_true', help="Add the lineage information.")
        # parent_parser.add_argument('--count_mse_loss', action='store_true', help="Use the count mse loss instead of ce loss.")
        # Settings for the off-set layer:
        parent_parser.add_argument('--weight_loss_by_count', type=str2bool, default="false", help="Weight loss of each sample by their counting not frequency")
        parent_parser.add_argument('--no_normalization_in_batch', action='store_true', help="Don't normalize the loss weight within the batch!!")
        parent_parser.add_argument('--zero_offset', action='store_true', help="Set the sequences distribution at offset as 0")
        parent_parser.add_argument('--offset_share_layer', type=int, default=-1, help="Use the hidden state at layer i to output the offset.")
        parent_parser.add_argument('--transformer_offset', action='store_true', help="Use another transformer NN to predict the offset.")
        # parent_parser.add_argument('--regression_loss', action='store_true', help="Use the regression loss instead of the MLE loss.")
        # parent_parser.add_argument('--normalize_time_a', type=int, default=1,  help="t = (t-b)/a")
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
        # else:
            # print((shift_labels != -100).sum(-1))
            # if not getattr(self.config, "output_token_losses", False):
                # loss = loss.sum(-1) # TODO: / (shift_labels != -100).sum(-1) # calculate the loss for each sample
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
            # inputs_embeds = self.model.transformer.wte(batch["input_ids"])
            inputs_embeds = inputs_embeds + lineage_embeds.unsqueeze(1)
        outputs = self.model(inputs_embeds = inputs_embeds, labels = batch["labels"], attention_mask = batch["attention_mask"], output_hidden_states=True)
        # outputs = self.model(input_ids = batch["input_ids"], labels = batch["labels"], attention_mask = batch["attention_mask"], output_hidden_states=True)
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

    def logistic_time_loss_weight(self, time):
        k = 0.1
        x0 = 50
        return 1 / (1 + torch.exp(-k * (time - x0)))

    def forward(self, batch, batch_idx, reduce=True, return_rate=False, return_offset=False, mode="train"):
        if getattr(self.config, "zero_time", False):
            batch["input_time"].fill_(0.)

        if getattr(self.config, "set_time", None) is not None:
            batch["input_time"].fill_(self.config.set_time) # set time bin as a constant.

        # if getattr(self.config, "ensemble", False):
        #     for _model in self.model:
        #         logits = self.model(**batch).logits

        logits = self.model(**batch).logits / self.config.temperature
        if self.config.weight_loss_by_count and batch.get('freq', None) is not None and batch.get('bin_size', None) is not None:
            loss_weight = batch.get('freq', None) * batch.get('bin_size', None)
        elif not self.config.weight_loss_by_count and batch.get('freq', None) is not None:
            loss_weight = batch.get('freq', None)
        else:
            loss_weight = 1.0
        
        if getattr(self.config, "weight_loss_by_time", False):
            loss_weight = loss_weight * self.logistic_time_loss_weight(batch["input_time"])

        labels = batch["labels"]

        loss = self.nll_loss(logits, labels, loss_weight=loss_weight, reduce=reduce)
        
        # loss_dict = {}
        # if return_rate:
        #     loss_dict["rate"] = self.get_rate(rate, labels)
        # if return_offset and not self.config.zero_offset:
        #     loss_dict["offset"] = self.get_rate(offset, labels)

        return loss, {}

    def training_step(self, batch, batch_idx):
        # self.generate("A")
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
        # print(loss.size())
        # print(loss)
        # token_num = torch.sum(batch["labels"][..., 1:].contiguous() != self.alphabet.pad(), dim=-1)
        token_num = torch.sum(
            (batch["labels"][..., 1:].contiguous() != self.alphabet.pad()) * 
            (batch["labels"][..., 1:].contiguous() != self.alphabet.eos()) * 
            (batch["labels"][..., 1:].contiguous() != self.alphabet.bos()), dim=-1)

        if "freq" in batch and "bin_size" in batch:
            weight = batch["freq"] * batch["bin_size"]
        else:
            weight = token_num.new_zeros(token_num.size(0)) + 1.0
        # print(token_num)
        # print(weight)
        # exit()
        # exit()
        self.log("test_loss", loss.mean(), prog_bar=True)
        # for key in loss_dict:
            # self.log("test_%s" % key, loss_dict[key].mean(), prog_bar=True)
        return loss, token_num, weight

    # def generate(self, text: str, device: torch.device = torch.device("cpu"), **kwargs) -> Any:
    #     # inputs = self.alphabet.encode_line("A")
    #     # print(inputs)
    #     inputs = inputs.to(self.model.device)
    #     input_time = torch.tensor([10.0]).to(self.model.device)
    #     # print(self.model)
    #     # print(self.model.generate(inputs.unsqueeze(0), input_time=input_time))
    #     # exit()
    #     return self.model.generate(inputs, **kwargs)

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
        generate_kwargs["num_return_sequences"] = getattr(self.config, "num_return_sequences", 1.0)
        if getattr(self.config, "generate_max_length", None) is None:
            generate_kwargs["max_length"] = self.config.max_position_embeddings
        else:
            generate_kwargs["max_length"] = getattr(self.config, "generate_max_length", None)
        generate_kwargs["pad_token_id"] = self.alphabet.pad()
        generate_kwargs["eos_token_id"] = self.alphabet.eos()
        generate_kwargs["bos_token_id"] = self.alphabet.bos()
        
        if batch["input_ids"][0, -1].item() == self.alphabet.eos():
            batch["input_ids"] = batch["input_ids"][:, :-1]

        # print(generate_kwargs)

        # model_inputs = {"input_ids": batch["input_ids"], "input_time": batch["input_time"]}
        # print(model_inputs)
        # print(self.model.tokenizer)
        # generate_kwargs["do_sample"] = False
        output_ids = self.model.generate(**batch, **generate_kwargs)
        # print(output_ids.size())
        input_time = batch["input_time"].unsqueeze(1).repeat(1, generate_kwargs["num_return_sequences"]).view(-1)
        # print(input_time.size())
        outputs = [{"prediction": self.alphabet.string(x), "src_time": input_time[i].item()} for i, x in enumerate(output_ids)]
        # print(outputs)
        # print(self.alphabet.bos(), self.alphabet.eos())
        return outputs
        
    def test_epoch_end(self, outputs):
        losses, token_nums, weights = [], [], []
        # print(len(outputs))
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
        # print("Sum of frequency", torch.exp(-losses).sum())
        print(torch.sum(weights), torch.sum(token_nums * weights))
        # print(losses.size(), token_nums.size(), weights.size())
        # print(outputs[0]) # loss, token_num, weight
        # ppl1 = torch.exp(torch.sum(losses) / torch.sum(token_nums))
        # print(ppl1)
        # ppl2 = torch.exp(torch.sum(weights * losses) / torch.sum(weights * token_nums))
        # print(ppl2)
        ppl = torch.exp(torch.sum(losses * weights) / torch.sum(token_nums * weights))
        nll = torch.sum(weights * losses) / torch.sum(weights)
        # nll = torch.exp(torch.sum(losses * weights))
        # exit()
        # collate data:
        # outputs is a list of dict, or a list of list of dict (for multiple dataloaders)
        # loss = torch.cat(outputs)
        self.log_dict({"perplexity": ppl, "nll": nll, "coverage": torch.exp(-losses).sum()})

        if self.config.output_token_losses:
            self.all_outputs = []
            for dataloader_outputs in outputs:
                for output in dataloader_outputs:
                    # print(output[0].size())
                    # loss = 
                    self.all_outputs.extend([x for x in output[0]])
        else:
            self.all_outputs = []
            # for loss in losses:
            #     # info_dict["prediction"] = loss.item()
            #     self.all_outputs.append(loss.item())
            for loss, tok_num in zip(losses, token_nums):
                # info_dict["prediction"] = loss.item()
                self.all_outputs.append({"prediction": loss.item(), "token_num": tok_num.item()})

        return ppl
    
    def output_testing_results(self, outputs, predict_dataset):
        
        predict_dataset = [item for sublist in predict_dataset for item in sublist]
        # print(len(outputs))
        # print(len(predict_dataset))
        assert len(outputs) == len(predict_dataset)
        results = []
        for index, output_loss in enumerate(outputs):
            # src_id,freq,src_time,prediction,rate,offset
            if self.config.output_token_losses:
                output_dict = {"prediction": " ".join([str(x.item()) for x in output_loss])}
            else:
                # output_dict = {"prediction": output_loss}
                output_dict = output_loss
            # print(output_dict)
            # exit()
            output_dict["src_id"] = predict_dataset[index]["src_id"]
            output_dict["src_time"] = predict_dataset[index]["src_time"]
            output_dict["freq"] = predict_dataset[index]["freq"]
            results.append(output_dict)
        return results

    def output_predicting_results(self, outputs, predict_dataset, *args, **kwargs):
        # assert len(outputs) == len(predict_dataset)
        # print(len(outputs), len(predict_dataset))
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

@register_model("gpt2_time_ensemble")
class GPT2TimeEnsemble(GPT2TimeNew):
    def __init__(self, *models) -> None:
        # super().__init__(con)
        # print(type(models), len(models))
        super().__init__(models[0].config, models[0].alphabet)
        self._models = nn.ModuleList(models)
        self.alphabet = models[0].alphabet
        self.config = models[0].config
    
    @classmethod
    def load_from_checkpoint(cls, paths, **args):
        paths = paths.split(",")
        return GPT2TimeEnsemble(*[GPT2TimeNew.load_from_checkpoint(path, **args) for path in paths])

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parent_parser = super(GPT2TimeEnsemble, cls).add_argparse_args(parent_parser)
        parent_parser.add_argument('--ensemble_checkpoints', type=str, nargs="+")
        return parent_parser

    def forward(self, batch, batch_idx, reduce=True, return_rate=False, return_offset=False):
        if getattr(self.config, "zero_time", False):
            batch["input_time"].fill_(0.)

        if getattr(self.config, "set_time", None) is not None:
            batch["input_time"].fill_(self.config.set_time) # set time bin as a constant.

        logits = []
        for model in self._models:
            _logits = model.model(**batch).logits
            _logits = nn.functional.log_softmax(_logits, dim=-1)
            # print(_logits.size())
            # print(torch.sum(torch.exp(_logits), dim=-1))
            logits.append(_logits)

        logits = torch.stack(logits, dim=0) / self.config.temperature
        # print(logits.size())
        logits = torch.logsumexp(logits, dim=0) - math.log(logits.size(0))
        # print(torch.sum(torch.exp(logits), dim=-1))
        loss_weight = batch.get('freq', None)

        labels = batch["labels"]

        loss = self.nll_loss(logits, labels, loss_weight=loss_weight, reduce=reduce)
        
        return loss, {}
