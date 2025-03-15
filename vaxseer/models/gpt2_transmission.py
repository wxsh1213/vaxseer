import torch
from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer
import transformers
import torch.nn as nn
from data.utils import discretize_time
from torch.nn import CrossEntropyLoss
from models import register_model
import math, logging
from typing import IO, Any, Callable, Dict, Optional, Union
from utils.args import str2bool
from transformers import AutoConfig
from collections import defaultdict, namedtuple
import os
from copy import deepcopy

GPTOutputs = namedtuple('GPTOutputs', ['logits', 'info_dict'])

class GPT2TimeTransmissionModule(transformers.GPT2LMHeadModel):
    def __init__(self, config, num_component, base_models=None, **args) -> None:
        self.global_logits_reg_w = getattr(config, "global_logits_reg_w", 0.0)

        super().__init__(config)
        self.num_component = num_component

        self.config = config
        self.eps = config.min_rate_value
        self.inf = config.max_rate_value

        if config.pos_function == "softplus":
            self.pos_func = torch.nn.Softplus()
        elif config.pos_function == "sigmoid":
            self.pos_func = torch.nn.Sigmoid()
        elif config.pos_function == "relu":
            self.pos_func = torch.nn.ReLU()
        elif config.pos_function == "exp":
            self.pos_func = torch.exp
        elif config.pos_function == "abs":
            self.pos_func = torch.abs
        else:
            self.pos_func = None

        if config.offset_pos_function == "softmax":
            self.offset_pos_func = nn.Softmax(dim=-1)
        elif config.offset_pos_function == "softplus":
            self.offset_pos_func = torch.nn.Softplus()
        elif config.offset_pos_function == "relu":
            self.offset_pos_func = torch.nn.ReLU()
        elif config.offset_pos_function == "abs":
            self.offset_pos_function = torch.abs
        else:
            self.offset_pos_func = None
        
        self.build_models(config, num_component, base_models=base_models, **args)
        
    def build_global_models(self, config, base_models):
        _model_config = deepcopy(config)
        setattr(_model_config, "data_property", 'global')
        self.global_evolution_model = GPT2TimeTransmissionSimpleModule(_model_config, 1, base_models=base_models)

    def build_models(self, config, num_component, base_models=None, **args):
        if base_models is not None:
            self.trans_base = base_models["trans_base"]
            self.offsets_base = base_models["offsets_base"]
        else:
            self.trans_base = transformers.GPT2LMHeadModel(config)
            if self.config.transformer_offset:
                self.offsets_base = transformers.GPT2LMHeadModel(config)
            else:
                self.offsets_base = self.trans_base
            
            base_models = {
                "trans_base": self.trans_base,
                "offsets_base": self.offsets_base
            }

        self.trans_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size) for _ in range(num_component * (num_component + 1) // 2)]) 
        self.offsets_heads = nn.ModuleList([nn.Linear(config.hidden_size, config.vocab_size) for _ in range(num_component)]) 
    
        if self.global_logits_reg_w:
            self.build_global_models(config, base_models)
    
    def get_initial_prob(self, input_ids, labels, attention_mask, cache_hidden_states=None):
        prob_vectors = []
        if cache_hidden_states is None:
            outputs = self.trans_base.forward(input_ids = input_ids, labels = labels, \
                    attention_mask = attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = cache_hidden_states

        for k in range(self.num_component):
            offset = self.offsets_heads[k](hidden_states)
            x0 = self.offset_pos_func(offset)
            prob_vectors.append(x0)

        prob_vectors = torch.stack(prob_vectors, dim=-1) # [B, L, V, K]
        return prob_vectors.view(-1, self.num_component)
        
    def get_trans_matrix_from_base(self, input_ids, labels, attention_mask, cache_hidden_states=None, generation=False, **args):

        if cache_hidden_states is None:
            outputs = self.trans_base.forward(input_ids = input_ids, labels = labels, \
                    attention_mask = attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = cache_hidden_states
        
        if generation:
            L = hidden_states.size(1) # Real len of sequences
            hidden_states = hidden_states[:, -1:, :]

        rates_matrix = [[0 for _ in range(self.num_component)] for _ in range(self.num_component)]
        for i in range(self.num_component):
            for j in range(i, self.num_component):
                k = ((2 * self.num_component - i + 1) * i) // 2 + j - i
                rate = self.trans_heads[k](hidden_states)
                rate = rate + torch.rand(rate.size()).to(rate.device) * self.eps # To avoid the A is ill-defined.
                rates_matrix[i][j] = rate
                rates_matrix[j][i] = rate
        rates_matrix = [item for row in rates_matrix for item in row]
        rates_matrix = torch.stack(rates_matrix, dim=-1) # [B, L, V, K**2]
        rates_matrix = rates_matrix.view(rates_matrix.size(0), rates_matrix.size(1), rates_matrix.size(2), self.num_component, self.num_component)
        rates_matrix = self.pos_func(rates_matrix)

        # to avoid overfloat
        rates_matrix = torch.clamp(rates_matrix, min=self.eps, max=self.inf)
        eig_value, eig_vector = torch.linalg.eigh(rates_matrix.view(-1, self.num_component, self.num_component))
        
        if getattr(self.config, "topk_eigen", None) is not None:
            eig_value = eig_value[:, -self.config.topk_eigen:]
            eig_vector = eig_vector[:, :, -self.config.topk_eigen:] # V[:,:,-2:]
        
        assert ~torch.any(torch.isnan(eig_value))
        assert ~torch.any(torch.isnan(eig_vector))

        if generation:
            eig_value = eig_value.view(-1, 1, self.config.vocab_size, eig_value.size(-1)).repeat(1, L, 1, 1).view(-1, eig_value.size(-1))
            eig_vector = eig_vector.view(-1, 1, self.config.vocab_size, eig_vector.size(-2), eig_vector.size(-1)).repeat(1, L, 1, 1, 1).view(-1, eig_vector.size(-2), eig_vector.size(-1))

        return  eig_value, eig_vector, rates_matrix

    @classmethod
    def from_config(cls, config, num_component, **args):
        model = cls(config, num_component, **args)
        return model

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs) -> Dict[str, Any]:
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method.
        """
        kwargs.pop("attention_mask")
        ret_dict = super().prepare_inputs_for_generation(input_ids, past, **kwargs)
        ret_dict["input_time"] = kwargs["input_time"]
        ret_dict[self.config.data_property] = kwargs[self.config.data_property]
        ret_dict["generation"] = True

        if self.config.num_beams > 1:
            # extend the input_time and 
            bsz = ret_dict[self.config.data_property].size(0)
            expanded_return_idx = (
                torch.arange(bsz).view(-1, 1).repeat(1, self.config.num_beams).view(-1).to(input_ids.device)
            )
            ret_dict[self.config.data_property] = ret_dict[self.config.data_property].index_select(0, expanded_return_idx)
            ret_dict["input_time"] = ret_dict["input_time"].index_select(0, expanded_return_idx)

        return ret_dict
    
    def forward_cache_hidden_states(self, input_time, **argv):
        return {} # Do nothing. But you could rewrite this if necessary.

    @property
    def number_of_component(self, ):
        return self.num_component

    def get_continent_evolution_outputs(self, input_time, trans_cache_hidden_states=None, offsets_cache_hidden_states=None, **argv):
        continent_outputs = self.continent_evolution_model.forward(input_time, **argv, trans_cache_hidden_states=trans_cache_hidden_states, \
                offsets_cache_hidden_states=offsets_cache_hidden_states, return_component_logits=True) # [B, L, V, 6]
        return continent_outputs

    def forward(self, input_time, **argv):
        generation = argv.pop("generation", False)

        return_rates_matrix = argv.get("return_rates_matrix", False)
        return_init_prob = argv.get("return_init_prob", False)
        return_eigen_values = argv.get("return_eigen_values", False)
        return_logits = argv.get("return_logits", False)
        return_component_logits = argv.get("return_component_logits", False)
        
        caches = self.forward_cache_hidden_states(input_time, **argv)
        trans_cache_hidden_states = caches.get("trans_cache_hidden_states")
        offsets_cache_hidden_states = caches.get("offsets_cache_hidden_states")
        if "trans_cache_hidden_states" in argv:
            trans_cache_hidden_states = argv.get("trans_cache_hidden_states")
        if "offsets_cache_hidden_states" in argv:
            offsets_cache_hidden_states = argv.get("offsets_cache_hidden_states")

        info_dict = {}

        time = discretize_time(
            input_time, 
            one_step=False, 
            normalize_time_a=self.config.normalize_time_a, 
            normalize_time_b=self.config.normalize_time_b,
            discrete=False)
        beam_size = argv.get("input_ids").size(0) // input_time.size(0)
        time = time.unsqueeze(1).repeat(1, beam_size).view(-1)

        B, L = argv.get("input_ids").size()
        V = self.config.vocab_size

        # [B*L*V, K], [B*L*V,K,K]
        eig_value, eig_vecs, rates_matrix = self.get_trans_matrix_from_base(
            argv.get("input_ids"), argv.get("labels"), argv.get("attention_mask"), 
            cache_hidden_states=trans_cache_hidden_states, generation=generation)

        # [B*L*V, K]
        init_prob = self.get_initial_prob(argv.get("input_ids"), argv.get("labels"), argv.get("attention_mask"), cache_hidden_states=offsets_cache_hidden_states)
        
        # eig_vecs: [B*L*V, K, K'], K' might <= K when only considering top-k eigenvalues
        const = torch.bmm(eig_vecs.transpose(-2, -1), init_prob.unsqueeze(-1)).squeeze(-1) # [B*L*V,K']

        time = time.reshape(-1, 1, 1).expand(-1, L, V).reshape(-1, 1) # [B*L*V,1]
        
        # 
        p = (const * torch.exp(time * eig_value)).unsqueeze(1) * eig_vecs # [B*L*V, K, K']
        p = torch.sum(p, dim=-1) # [B*L*V, K]
        # 

        p = p.view(B, -1, self.number_of_component) #[B, L*V, K]
        p = torch.clamp(p, min=self.eps, max=self.inf) # For numerical stable...
        
        if return_component_logits:
            info_dict["component_logits"] = torch.log(p).view(B, L, V, -1)  # 
            
        if self.config.data_property in argv:
            host_label = argv[self.config.data_property].unsqueeze(1).repeat(1, p.size(1)).unsqueeze(-1).long() # [B, L*V, 1]
            _p = torch.gather(p, -1, host_label).squeeze(-1).view(B, L, -1) # [B, L, V]
            logits = torch.log(_p)
            
            if return_rates_matrix:
                info_dict["rates_matrix"] = rates_matrix.view(B, L, V, self.num_component, self.num_component)

            if return_init_prob:
                init_prob_ = init_prob.view(B, L, V, self.num_component) # [B, L, V, K]
                info_dict["init_prob"] = init_prob_

        else:
            logits = None
        
        if self.global_logits_reg_w > 0:
            argv["global"] = torch.zeros(argv[self.config.data_property].size(), device=argv[self.config.data_property].device)
            global_outputs = self.global_evolution_model.forward(input_time, **argv, trans_cache_hidden_states=trans_cache_hidden_states, \
                    offsets_cache_hidden_states=offsets_cache_hidden_states) # [B, L, V, 6]
            info_dict["global_logits"] = global_outputs.logits
            info_dict["sum_local_logits"] = torch.logsumexp(torch.log(p).view(B, L, V, -1), dim=-1)  # , dim=-1)
        
        return GPTOutputs(logits=logits, info_dict=info_dict)

class GPT2TimeTransmissionSimpleModule(GPT2TimeTransmissionModule):

    def __init__(self, config, num_component, base_models=None, **args) -> None:
        super().__init__(config, num_component, base_models, **args)
        self.offset_pos_func = nn.Softmax(dim=-2)

    def build_models(self, config, num_component, base_models=None, **args):
        if base_models is not None:
            self.trans_base = base_models["trans_base"]
            if self.config.transformer_offset:
                self.offsets_base = base_models["offsets_base"]
            else:
                self.offsets_base = self.trans_base
        else:
            self.trans_base = transformers.GPT2LMHeadModel(config)
            if self.config.transformer_offset:
                self.offsets_base = transformers.GPT2LMHeadModel(config)
            else:
                self.offsets_base = self.trans_base

        self.eigvecs_heads = nn.Linear(config.hidden_size, config.vocab_size * num_component * num_component)
        self.eigvals_heads = nn.Linear(config.hidden_size, config.vocab_size * num_component)
    
    def get_trans_matrix_from_base(self, input_ids, labels, attention_mask, cache_hidden_states=None):
        if cache_hidden_states is None:
            outputs = self.trans_base.forward(input_ids = input_ids, labels = labels, \
                    attention_mask = attention_mask, output_hidden_states=True)
            hidden_states_eigval = outputs.hidden_states[-1]

            if self.config.transformer_offset:
                outputs = self.offsets_base.forward(input_ids = input_ids, labels = labels, \
                    attention_mask = attention_mask, output_hidden_states=True)
                hidden_states_eigvec = outputs.hidden_states[-1]
            else:
                hidden_states_eigvec = hidden_states_eigval
        else:
            hidden_states_eigval = cache_hidden_states
            hidden_states_eigvec = cache_hidden_states

        eig_value = self.eigvals_heads(hidden_states_eigval).view(-1, self.num_component)
        eig_vector = self.eigvecs_heads(hidden_states_eigvec).view(-1, self.num_component, self.num_component)

        return  eig_value, eig_vector, None

    def forward(self, input_time, **argv):
        return_component_logits = argv.get("return_component_logits", False)
        
        caches = self.forward_cache_hidden_states(input_time, **argv)
        trans_cache_hidden_states = caches.get("trans_cache_hidden_states")
        offsets_cache_hidden_states = caches.get("offsets_cache_hidden_states")
        # Overwrite
        if "trans_cache_hidden_states" in argv:
            trans_cache_hidden_states = argv.get("trans_cache_hidden_states")
        if "offsets_cache_hidden_states" in argv:
            offsets_cache_hidden_states = argv.get("offsets_cache_hidden_states")

        info_dict = {}

        time = discretize_time(
            input_time, 
            one_step=False, 
            normalize_time_a=self.config.normalize_time_a, 
            normalize_time_b=self.config.normalize_time_b,
            discrete=False)
        beam_size = argv.get("input_ids").size(0) // input_time.size(0)
        time = time.unsqueeze(1).repeat(1, beam_size).view(-1)

        B, L = argv.get("input_ids").size()
        V = self.config.vocab_size

        eig_value, eig_vecs, _ = self.get_trans_matrix_from_base(argv.get("input_ids"), argv.get("labels"), argv.get("attention_mask"), cache_hidden_states=trans_cache_hidden_states)

        time = time.reshape(-1, 1, 1).expand(-1, L, V).reshape(-1, 1) # [B*L*V,1]
        
        logits = (time * eig_value).unsqueeze(1) + eig_vecs # [B*L*V, K, K]
        logits = torch.logsumexp(logits, dim=-1)
        if return_component_logits:
            info_dict["component_logits"] = logits.view(B, L, V, -1)
                
        logits = logits.view(B, -1, self.number_of_component) # [B, L*V, K]
        host_label = argv[self.config.data_property].unsqueeze(1).repeat(1, logits.size(1)).unsqueeze(-1).long() # [B, L*V, 1]
        logits = torch.gather(logits, -1, host_label).squeeze(-1).view(B, L, -1) # [B, L, V]

        return GPTOutputs(logits=logits, info_dict=info_dict)

class GPT2TimeTransmissionParamShareModule(transformers.GPT2LMHeadModel):
    def __init__(self, config, num_component, base_models=None, **args) -> None:
        super().__init__(config)
        self.num_component = num_component
        if base_models:
            self.base_rate = base_models["trans_base"]
            self.base_offset = base_models["offsets_base"]
        else:
            self.base_rate = transformers.GPT2LMHeadModel(config)

            if config.transformer_offset:
                self.base_offset = transformers.GPT2LMHeadModel(config)
            else:
                self.base_offset = self.base_rate

        self.rate_output_heads = nn.Linear(config.hidden_size, config.vocab_size * num_component)
        self.offset_output_heads = nn.Linear(config.hidden_size, config.vocab_size * num_component)

    @classmethod
    def from_config(cls, config, num_component):
        model = cls(config, num_component)
        return model
    
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs) -> Dict[str, Any]:
        kwargs.pop("attention_mask")
        ret_dict = super().prepare_inputs_for_generation(input_ids, past, **kwargs)
        ret_dict["input_time"] = kwargs["input_time"]
        ret_dict[self.config.data_property] = kwargs[self.config.data_property]
        ret_dict["generation"] = True

        if self.config.num_beams > 1:
            bsz = ret_dict[self.config.data_property].size(0)
            expanded_return_idx = (
                torch.arange(bsz).view(-1, 1).repeat(1, self.config.num_beams).view(-1).to(input_ids.device)
            )
            ret_dict[self.config.data_property] = ret_dict[self.config.data_property].index_select(0, expanded_return_idx)
            ret_dict["input_time"] = ret_dict["input_time"].index_select(0, expanded_return_idx)

        return ret_dict
            
    def get_rates(self, **argv):
        trans_cache_hidden_states = argv.get("trans_cache_hidden_states")
        if trans_cache_hidden_states is None:
            outputs = self.base_rate.forward(input_ids = argv.get("input_ids"), labels = argv.get("labels"), \
                attention_mask = argv.get("attention_mask"), output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = trans_cache_hidden_states
        
        B, L = hidden_states.size(0), hidden_states.size(1)        
        logits = self.rate_output_heads(hidden_states).view(B, L, -1, self.num_component) # [B, L, V*K]
        label = argv[self.config.data_property].view(-1, 1, 1, 1).repeat(1, logits.size(1), logits.size(2), 1).long() # [B, L, V, 1]
        logits_reduce = torch.gather(logits, -1, label).squeeze(-1) # [B, L, V]
        return logits_reduce, logits
    
    def get_offsets(self, **argv):
        offsets_cache_hidden_states = argv.get("offsets_cache_hidden_states")
        if offsets_cache_hidden_states is None:
            outputs = self.base_offset.forward(input_ids = argv.get("input_ids"), labels = argv.get("labels"), \
                attention_mask = argv.get("attention_mask"), output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = offsets_cache_hidden_states
        B, L = hidden_states.size(0), hidden_states.size(1)
        logits = self.offset_output_heads(hidden_states).view(B, L, -1, self.num_component)# [B, L, V*K ]
        host_label = argv[self.config.data_property].view(-1, 1, 1, 1).repeat(1, logits.size(1), logits.size(2), 1).long() # [B, L, V, 1]
        logits_reduce = torch.gather(logits, -1, host_label).squeeze(-1) # [B, L, V]
        return logits_reduce, logits

    def forward(self, input_time, **argv):
        time = discretize_time(
            input_time, 
            one_step=False, 
            normalize_time_a=self.config.normalize_time_a, 
            normalize_time_b=self.config.normalize_time_b,
            discrete=False)
        beam_size = argv.get("input_ids").size(0) // input_time.size(0)
        time = time.unsqueeze(1).repeat(1, beam_size).view(-1)
        reduce_rate, rate = self.get_rates(**argv)
        reduce_offset, offset = self.get_rates(**argv)
        logits_reduce = reduce_rate * time.unsqueeze(-1).unsqueeze(-1) + reduce_offset
        logits_full = rate * time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + offset
        return GPTOutputs(
            logits=logits_reduce, 
            info_dict={"component_logits": logits_full} if argv.get("return_component_logits", False) else {})

@register_model("gpt2_time_transmission")
class GPT2TimeTransmission(LanguageModelingTransformer):
    def __init__(self, config, alphabet) -> None:
        self.config = config
        self.alphabet = alphabet
        self.pad_idx = alphabet.pad()
        super().__init__(
            pretrained_model_name_or_path="gpt2",
            load_weights=False,
            vocab_size=len(alphabet),
            max_position_embeddings=config.max_position_embeddings,
            num_hidden_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size
            )
        
    def set_transformer_config(self, config):
        setattr(config, "num_hidden_layers", self.config.num_hidden_layers)
        setattr(config, "hidden_size", self.config.hidden_size)
        setattr(config, "normalize_time_a", self.config.normalize_time_a)
        setattr(config, "normalize_time_b", self.config.normalize_time_b)
        setattr(config, "transformer_offset", self.config.transformer_offset)
        setattr(config, "data_property", self.config.data_properties[0])
        setattr(config, "offset_pos_function", getattr(self.config, "offset_pos_function", "softmax"))
        setattr(config, "pos_function", getattr(self.config, "pos_function", "softplus"))
        setattr(config, "max_rate_value", getattr(self.config, "max_rate_value", 1e5))
        setattr(config, "min_rate_value", getattr(self.config, "min_rate_value", 1e-12))
        setattr(config, "padding_idx", self.alphabet.pad())
        setattr(config, "data_properties", self.config.data_properties)
        for data_property in self.config.data_properties:
            setattr(config, "%s_dict" % data_property, getattr(self.config, "%s_dict" % data_property))
        setattr(config, "topk_eigen", getattr(self.config, "topk_eigen", None))
        setattr(config, "global_logits_reg_w", getattr(self.config, "global_logits_reg_w", 0))

    def initialize_model(self, pretrained_model_name_or_path: str):
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path="gpt2", **self.model_data_kwargs
        )
        self.set_transformer_config(config)

        if self.config.num_host:
            num_host = self.config.num_host
        else:
            num_host = len(getattr(self.config, "%s_dict" % self.config.data_properties[0]))
        
        logging.info("num_host: %d" % num_host)

        output_layer_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path="gpt2", 
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers
        )
        
        if self.config.implement_version == 1:
            self.model = GPT2TimeTransmissionModule.from_config(config, num_host, output_layer_config=output_layer_config)
        elif self.config.implement_version == 3:
            assert len(self.config.data_properties) == 1, "Could only set one property!"
            self.model = GPT2TimeTransmissionParamShareModule.from_config(config, num_host)
        
    def configure_optimizers(self) -> Dict:
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
        args = None,
        **kwargs
    ):
        model = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict)
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
        parent_parser.add_argument('--num_hidden_layers', type=int, default=12)
        parent_parser.add_argument('--hidden_size', type=int, default=768)
        parent_parser.add_argument('--model_name_or_path', type=str, default="gpt2")
        parent_parser.add_argument('--normalize_time_a', type=int, default=1,  help="Normalize the time: t = (t-b)/a")
        parent_parser.add_argument('--normalize_time_b', type=int, default=0, help="Normalize the time: t = (t-b)/a")
        parent_parser.add_argument('--weight_loss_by_count', type=str2bool, default="true", help="Weight loss of each sample by their counting not frequency")
        parent_parser.add_argument('--transformer_offset', action='store_true', help="Use another transformer to predict the offset.")
        parent_parser.add_argument('--output_token_losses', type=str2bool, default="false")

        # generation configs
        parent_parser.add_argument('--do_sample', type=str2bool, default="false")
        parent_parser.add_argument('--temperature', type=float, default=1.0)
        parent_parser.add_argument('--num_beams', type=int, default=1)
        parent_parser.add_argument('--num_return_sequences', type=int, default=1)
    
        parent_parser.add_argument('--num_host', type=int, default=None, help="Number of sub-populations.")
        parent_parser.add_argument('--implement_version', type=int, default=1, choices=[1, 3])
        parent_parser.add_argument('--return_rates_matrix', type=str2bool, default="false")
        parent_parser.add_argument('--return_init_prob', type=str2bool, default="false")
        parent_parser.add_argument('--offset_pos_function', type=str, default="softmax", choices=["softmax", "softplus", "relu", "none", "exp", "abs"])
        parent_parser.add_argument('--pos_function', type=str, default="softplus", choices=["sigmoid", "softplus", "relu", "none", "exp", "abs"])
        parent_parser.add_argument('--max_rate_value', type=float, default=1e5)
        parent_parser.add_argument('--min_rate_value', type=float, default=1e-12)
        parent_parser.add_argument('--topk_eigen', type=int, default=None)
        parent_parser.add_argument('--global_logits_reg_w', type=float, default=0.0)
        return parent_parser
        
    def nll_loss(self, lm_logits, labels, loss_weight=None, reduce=True, ignore_bos=False):
        labels = labels.masked_fill(torch.eq(labels, self.alphabet.pad()), -100)
        if ignore_bos:
            labels = labels.masked_fill(torch.eq(labels, self.alphabet.bos()), -100)
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous() / self.config.temperature
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduce=False)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())

        if reduce:
            loss = loss.sum(dim=-1) / (shift_labels != -100).sum(dim=-1) # [B]
            if loss_weight is not None:
                loss_weight = loss_weight / loss_weight.sum()
                loss = torch.sum(loss * loss_weight)
            else:
                loss = loss.mean()
        return loss

    def forward(self, batch, batch_idx, reduce=True, mode="train"):

        if self.config.weight_loss_by_count and batch.get('freq', None) is not None and batch.get('bin_size', None) is not None:
            loss_weight = batch.get('freq', None) * batch.get('bin_size', None)
        elif not self.config.weight_loss_by_count and batch.get('freq', None) is not None: # otherwise, using the frequency
            loss_weight = batch.get('freq', None)
        else:
            loss_weight = 1.0
        labels = batch["labels"]
        model_outputs = self.model(
            **batch, return_rates_matrix=self.config.return_rates_matrix, 
            return_init_prob = self.config.return_init_prob, return_component_logits=True,
            )
        loss = self.nll_loss(model_outputs.logits, labels, loss_weight=loss_weight, reduce=reduce, 
                            ignore_bos=True if mode == "test" else False)
                
        loss_dict = {}
        if getattr(self.config, "global_logits_reg_w", 0) > 0 and mode != 'test':
            global_logits = model_outputs.info_dict["global_logits"]
            sum_local_logits = model_outputs.info_dict["sum_local_logits"]
            B = global_logits.size(0)
            assert global_logits.size() == sum_local_logits.size()
            global_reg_loss = torch.mean((global_logits.view(B, -1) - sum_local_logits.view(B, -1)) ** 2, dim=-1) # [B]
            if loss_weight is not None:
                loss_weight = loss_weight / loss_weight.sum()
                global_reg_loss = torch.sum(global_reg_loss * loss_weight)
            else:
                global_reg_loss = global_reg_loss.mean()
            loss_dict["global_reg_loss"] = global_reg_loss
            loss = loss + self.config.global_logits_reg_w * global_reg_loss

        if self.config.return_rates_matrix:
            loss_dict["rates_matrix"] = model_outputs.info_dict["rates_matrix"]
            loss_dict["input_ids"] = batch["input_ids"]
        if self.config.return_init_prob:
            loss_dict["init_prob"] = model_outputs.info_dict["init_prob"]

        return loss, loss_dict

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
            (batch["labels"][..., 1:].contiguous() != self.alphabet.pad()) * (batch["labels"][..., 1:].contiguous() != self.alphabet.eos()), dim=-1)
        if "freq" in batch and "bin_size" in batch:
            weight = batch["freq"] * batch["bin_size"]
        else:
            weight = token_num.new_zeros(token_num.size(0)) + 1.0
        self.log("test_loss", loss.mean(), prog_bar=True)
        return loss, token_num, weight, loss_dict

    def overwrite_generate_kwargs(self, new_config):
        setattr(self.config, "do_sample", new_config.do_sample)
        setattr(self.config, "num_beams", new_config.num_beams)
        setattr(self.config, "temperature", new_config.temperature)
        setattr(self.config, "num_return_sequences", new_config.num_return_sequences)
        setattr(self.config, "output_token_losses", new_config.output_token_losses)
        setattr(self.config, "return_rates_matrix", new_config.return_rates_matrix)
        setattr(self.config, "return_init_prob", new_config.return_init_prob)

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
        other_info = defaultdict(list)
        if len(self.config.test_data_paths) == 1:
            outputs = [outputs]

        for dataloader_outputs in outputs:
            for output in dataloader_outputs:
                losses.append(output[0].sum(-1))
                token_nums.append(output[1])
                weights.append(output[2])
                
                for key in output[3]:
                    if isinstance(output[3][key], list) and isinstance(output[3][key][0], torch.Tensor):
                        other_info[key].extend([x.mean(dim=0).cpu() for x in output[3][key]])
                    
                    elif isinstance(output[3][key], torch.Tensor):
                        other_info[key].append(output[3][key].cpu())

        losses = torch.cat(losses)
        token_nums = torch.cat(token_nums)
        weights = torch.cat(weights)

        ppl = torch.exp(torch.sum(losses * weights) / torch.sum(token_nums * weights))
        nll = torch.sum(weights * losses) / torch.sum(weights)

        # self.log_dict({"perplexity": ppl, "nll": nll, "coverage": torch.exp(-losses).sum()})

        if self.config.output_token_losses:
            self.all_outputs = []
            for dataloader_outputs in outputs:
                for output in dataloader_outputs:
                    self.all_outputs.extend([x for x in output[0]])
        else:
            self.all_outputs = []
            for loss, tok_num in zip(losses, token_nums):
                self.all_outputs.append({"prediction": loss.item(), "token_num": tok_num.item()})

        # Output the other information
        if not os.path.exists(self.trainer.logger.log_dir):
            os.makedirs(self.trainer.logger.log_dir)
        for key in other_info:
            output_path = os.path.join(self.trainer.logger.log_dir, "%s.pkl" % key)
            logging.info("Saving %s to %s" % (key, output_path))
            torch.save(other_info[key], output_path)
            
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
                output_dict = output_loss # {"prediction": output_loss}
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

        output_path = args[0]
        if output_path is not None and output_path.endswith(".csv"):
            fasta_path = output_path.split(".csv")[0] + ".fasta"
            logging.info("Writing generations to %s" % fasta_path)
            with open(fasta_path, "w") as fout:
                for i, data in enumerate(results):
                    fout.write(">%d\n%s\n\n" % (i, data["prediction"]))

        return results