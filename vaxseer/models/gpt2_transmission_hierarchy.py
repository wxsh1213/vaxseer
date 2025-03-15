import torch
import transformers
import torch.nn as nn
from models import register_model
import math, logging
from typing import IO, Any, Callable, Dict, Optional, Union
from utils.args import str2bool
from transformers import AutoConfig
from copy import deepcopy
import torch.nn.functional as F
from models.gpt2_transmission import GPTOutputs, GPT2TimeTransmission, GPT2TimeTransmissionModule, GPT2TimeTransmissionSimpleModule, GPT2TimeTransmissionParamShareModule

class GPT2TimeModelMultiHostsHierarchyBase(transformers.GPT2LMHeadModel):
    def __init__(self, config, num_components, **args) -> None:
        super().__init__(config)

        self.build_base_model(config, num_components)
        self.build_local_models(config, num_components)
        self.build_block_trans_models(config, num_components)
        self.build_other_models(config, num_components)
    
    @classmethod
    def from_config(cls, config, num_components, **args):
        model = cls(config, num_components, **args)
        return model

    def build_base_model(self, config, num_components):
        raise NotImplementedError()

    def build_other_models(self, config, num_components):
        # For example, approximation models
        raise NotImplementedError()

    def build_local_models(self, config, num_components):
        self.local_models = nn.ModuleList([GPT2TimeTransmissionModule(config, num_component) for num_component in num_components])

    def build_block_trans_models(self, config, num_components):
        raise NotImplementedError

    def get_local_outputs(self, input_time, *args, **argv):
        raise NotImplementedError()

    def get_block_trans_outputs(self, input_time, *args, **argv):
        raise NotImplementedError

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        """
        Implement in subclasses of [`PreTrainedModel`] for custom behavior to prepare inputs in the generate method.
        """
        inputs = {"input_ids": input_ids, "input_time": kwargs["input_time"]}
        for prop in self.config.data_properties:
            inputs[prop] = kwargs[prop]
        
        inputs["generation"]=True

        if self.config.num_beams > 1:
            bsz = inputs[self.config.data_properties[0]].size(0)
            expanded_return_idx = (
                torch.arange(bsz).view(-1, 1).repeat(1, self.config.num_beams).view(-1).to(input_ids.device)
            )
            for prop in self.config.data_properties:
                inputs[prop] = inputs[prop].index_select(0, expanded_return_idx)
            inputs["input_time"] = inputs["input_time"].index_select(0, expanded_return_idx)
        return inputs
  
    def get_country_long_index(self, countries, continents):
        num_countries_in_continents = torch.tensor([0] + self.num_components[:-1]).to(countries.device)
        country_long_index = countries + torch.cumsum(num_countries_in_continents, dim=0)[continents] # country_index
        return country_long_index

    def forward(self, input_time, **argv):
        local_output = self.get_local_outputs(input_time, **argv)
        block_transoutput = self.get_block_trans_outputs(input_time, **argv)
        return GPTOutputs(logits=local_output.logits + block_transoutput.logits)
        
class GPT2TimeTransmissionHierarchyModule(GPT2TimeModelMultiHostsHierarchyBase):
    def __init__(self, config, num_components, **args) -> None:
        self.num_components = num_components
        if config.implement_version == 1:
            self._model_class = GPT2TimeTransmissionModule # .from_config(config, self.config.num_host, output_layer_config=output_layer_config)
        elif config.implement_version == 3:
            self._model_class = GPT2TimeTransmissionParamShareModule # .from_config(config, self.config.num_host)
        else:
            raise NotImplementedError
        super().__init__(config, num_components, **args)

        self.trans_w_pos_function = config.trans_w_pos_function
        if config.trans_w_pos_function == "softplus":
            self.pos_func = torch.nn.Softplus()
        elif config.trans_w_pos_function == "relu":
            self.pos_func = torch.nn.ReLU()
        elif config.trans_w_pos_function == "sigmoid":
            self.pos_func = torch.nn.Sigmoid()
        elif config.trans_w_pos_function == "none":
            self.pos_func = lambda x: x
        
        self.eps = 1e-12

    def build_country_to_continent_indexing_matrix(self, num_components):
        total_country_number = sum(num_components)
        total_continent_number = len(num_components)
        num_countries_in_continents = torch.tensor([0] + self.num_components[:-1])
        num_countries_in_continents_cumsum = torch.cumsum(num_countries_in_continents, dim=0)
        m = torch.zeros(total_continent_number, total_country_number)
        for i, c in enumerate(num_components):
            m[i, num_countries_in_continents_cumsum[i]:num_countries_in_continents_cumsum[i]+c] = 1.0
        return m

    def build_base_model(self, config, num_components):
        if config.transformer_offset:
            self.base_models = {
                "trans_base": transformers.GPT2LMHeadModel(config),
                "offsets_base": transformers.GPT2LMHeadModel(config),
                }
        else:
            self._base = transformers.GPT2LMHeadModel(config)
            self.base_models = {
                "trans_base": self._base,
                "offsets_base": self._base,
                }

    def build_local_models(self, config, num_components):
        country_model_config = deepcopy(config)
        setattr(country_model_config, "data_property", config.data_properties[1])
        self.local_models = nn.ModuleList([self._model_class(country_model_config, num_component, base_models=self.base_models) for num_component in num_components])

    def build_block_trans_models(self, config, num_components):
        if not self.config.reuse_transformer_for_cross_block_trans:
            config_copy = deepcopy(config)
            config_copy.n_layer = config.block_trans_model_n_layer
            self.cross_block_trans_base_model = transformers.GPT2LMHeadModel(config_copy)

        total_country_number = sum(num_components)
        total_continent_number = len(num_components)

        D = config.vocab_size
        if self.config.block_trans_model == "country_to_continent":
            if self.config.use_linear_init_trans_weight:
                _country_to_continent_trans_weight = torch.rand(total_country_number, config.hidden_size, D * len(num_components))
                _country_to_continent_trans_weight = (_country_to_continent_trans_weight - 0.5) * 2 / math.sqrt(config.hidden_size)
                _country_to_continent_trans_bias = torch.rand(total_country_number, D * len(num_components))
                _country_to_continent_trans_bias = (_country_to_continent_trans_bias - 0.5) * 2 / math.sqrt(config.hidden_size)
            else:
                _country_to_continent_trans_weight = torch.randn(total_country_number, config.hidden_size, D * len(num_components))
                _country_to_continent_trans_bias = torch.randn(total_country_number, D * len(num_components))
            self.country_to_continent_trans_weight = nn.Parameter(_country_to_continent_trans_weight, requires_grad=True)
            self.country_to_continent_trans_bias = nn.Parameter(_country_to_continent_trans_bias, requires_grad=True)
        elif self.config.block_trans_model == "continent_to_continent":
            if self.config.use_linear_init_trans_weight:
                _continent_to_continent_trans_weight = torch.rand(total_continent_number, config.hidden_size, D * len(num_components))
                _continent_to_continent_trans_weight = (_continent_to_continent_trans_weight - 0.5) * 2 / math.sqrt(config.hidden_size)
                _continent_to_continent_trans_bias = torch.rand(total_continent_number, D * len(num_components))
                _continent_to_continent_trans_bias = (_continent_to_continent_trans_bias - 0.5) * 2 / math.sqrt(config.hidden_size)
            
                _continent_to_country_trans_weight = torch.rand(total_country_number, config.hidden_size, D)
                _continent_to_country_trans_weight = (_continent_to_country_trans_weight - 0.5) * 2 / math.sqrt(config.hidden_size)
                _continent_to_country_trans_bias = torch.rand(total_country_number, D)
                _continent_to_country_trans_bias = (_continent_to_country_trans_bias - 0.5) * 2 / math.sqrt(config.hidden_size)
            else:
                _continent_to_continent_trans_weight = torch.randn(total_continent_number, config.hidden_size, D * len(num_components))
                _continent_to_continent_trans_bias = torch.randn(total_continent_number, D * len(num_components))

                _continent_to_country_trans_weight = torch.randn(total_country_number, config.hidden_size, D)
                _continent_to_country_trans_bias = torch.randn(total_country_number, D)
            
            self.continent_to_continent_trans_weight = nn.Parameter(_continent_to_continent_trans_weight, requires_grad=True)
            self.continent_to_continent_trans_bias = nn.Parameter(_continent_to_continent_trans_bias, requires_grad=True)
            self.continent_to_country_trans_weight = nn.Parameter(_continent_to_country_trans_weight, requires_grad=True)
            self.continent_to_country_trans_bias = nn.Parameter(_continent_to_country_trans_bias, requires_grad=True)
        
    def block_trans_continent_to_continent(self, hidden_states, continent, country, continent_logits, return_trans_weight=False):
        V = self.config.vocab_size
        B, L = hidden_states.size(0), hidden_states.size(1)
        num_countries_in_continents = torch.tensor([0] + self.num_components[:-1]).to(hidden_states.device)
        country_long_index = country + torch.cumsum(num_countries_in_continents, dim=0)[continent]
        _cross_continents_trans_weight = torch.bmm(hidden_states, self.continent_to_continent_trans_weight[continent]) + self.continent_to_continent_trans_bias[continent].unsqueeze(1) # [B, L, V, 6]
        _continents_to_country_trans_weight = torch.bmm(hidden_states, self.continent_to_country_trans_weight[country_long_index]) + self.continent_to_country_trans_bias[country_long_index].unsqueeze(1) # [B, L, V]        
        _trans_weight = _cross_continents_trans_weight.view(B, L, V, -1) + _continents_to_country_trans_weight.view(B, L, V, 1)
        _trans_weight = self.pos_func(_trans_weight) # Make sure it is positive
        continent_mask = 1.0 - F.one_hot(continent, num_classes=len(self.num_components))
        continent_mask = continent_mask.to(hidden_states.device) #[B, 6]
        cross_continent_logits = torch.sum((continent_logits * _trans_weight) * continent_mask.view(B, 1, 1, -1), dim=-1) # [B, L, V]
        if return_trans_weight:
            info = {"continent_to_continent": _cross_continents_trans_weight.view(B, L, V, -1), 
                    "continent_to_country": _continents_to_country_trans_weight, 
                    "trans_weight": _trans_weight,
                    "cross_group_trans_weight": _trans_weight * continent_mask.view(B, 1, 1, -1)}
            return cross_continent_logits, info
        
        return cross_continent_logits, {"cross_group_trans_weight": _trans_weight * continent_mask.view(B, 1, 1, -1)}
    
    def get_intra_group_masks(self, ):
        mask = torch.ones((self.total_country_number, self.total_country_number))
        for i, number_of_countries in enumerate(self.num_components):
            offset_index = sum(self.num_components[:i])
            mask[offset_index:offset_index+number_of_countries, offset_index:offset_index+number_of_countries] = 0
        return mask

    def block_trans_country_to_continent(self, hidden_states, continent, country, continent_logits, return_trans_weight=False):
        V = self.config.vocab_size
        B, L = hidden_states.size(0), hidden_states.size(1)
        num_countries_in_continents = torch.tensor([0] + self.num_components[:-1]).to(hidden_states.device)
        country_long_index = country + torch.cumsum(num_countries_in_continents, dim=0)[continent]
        _trans_weight = torch.bmm(hidden_states, self.country_to_continent_trans_weight[country_long_index]) + self.country_to_continent_trans_bias[country_long_index].unsqueeze(1) # [B, L, V, 6]
        _trans_weight = self.pos_func(_trans_weight) # Make sure it is possitive
        _trans_weight = _trans_weight.view(B, L, -1, len(self.num_components)) # [B, L, V, 6] or [B, L, 1, 6]

        continent_mask = 1.0 - F.one_hot(continent, num_classes=len(self.num_components))
        continent_mask = continent_mask.to(hidden_states.device) #[B, 6]
        
        cross_continent_logits = torch.sum((continent_logits * _trans_weight) * continent_mask.view(B, 1, 1, -1), dim=-1) # [B, L, V]

        if return_trans_weight:
            return cross_continent_logits, {"trans_weight": _trans_weight,"cross_group_trans_weight": _trans_weight * continent_mask.view(B, 1, 1, -1)}
        return cross_continent_logits, {"cross_group_trans_weight": _trans_weight * continent_mask.view(B, 1, 1, -1)}

    def build_other_models(self, config, num_components):
        num_component_continent = len(num_components) # 6
        if config.continent_share_base_models:
            base_models = self.base_models
        else:
            base_models = None

        _model_config = deepcopy(config)
        setattr(_model_config, "data_property", config.data_properties[0]) # should be continent

        if self.config.use_simple_continent_model:
            self.continent_evolution_model = GPT2TimeTransmissionSimpleModule(_model_config, num_component_continent, base_models=base_models)
        else:
            self.continent_evolution_model = self._model_class(_model_config, num_component_continent, base_models=base_models)

    def _split_batch(self, mask, input_time, **argv):
        new_args = []
        new_argv = {}
        
        for key in argv:
            if isinstance(argv[key], torch.Tensor):
                new_argv[key] = argv[key][mask]
            else:
                new_argv[key] = argv[key]

        return input_time[mask], new_argv

    def get_continent_evolution_outputs(self, input_time, trans_cache_hidden_states=None, offsets_cache_hidden_states=None, **argv):
        if self.config.continent_share_base_models:
            continent_outputs = self.continent_evolution_model.forward(input_time, **argv, trans_cache_hidden_states=trans_cache_hidden_states, \
                    offsets_cache_hidden_states=offsets_cache_hidden_states, return_component_logits=True) # [B, L, V, 6]
        else:
            continent_outputs = self.continent_evolution_model.forward(input_time, **argv, return_component_logits=True) # [B, L, V, 6]
            
        return continent_outputs

    def cross_block_trans_base_model_forward(self, argv, trans_cache_hidden_states):
        
        if self.config.block_trans_model == "prepend":
            input_ids = argv["input_ids"] # [B, L]
            continents = argv.get(self.config.data_properties[0]) # B
            countries = argv.get(self.config.data_properties[1]) # B
            prepend_input_ids = torch.cat([continents.unsqueeze(1), countries.unsqueeze(1), input_ids], dim=1) # [B, L+2]
            # print(input_ids.size(), prepend_input_ids, prepend_input_ids.size())
            
            if argv.get("attention_mask") is not None:
                attention_mask = argv.get("attention_mask")
                attention_mask_prepend = attention_mask.new_ones(attention_mask.size(0), 2)
                attention_mask = torch.cat([attention_mask_prepend, attention_mask], dim=1) # [B, L + 2]
                # print(attention_mask.size())
            else:
                attention_mask = argv.get("attention_mask")

            cross_block_hidden_states = self.cross_block_trans_base_model.forward(input_ids = prepend_input_ids, 
                                                                                  labels = prepend_input_ids, 
                                                                                  attention_mask = attention_mask, 
                                                                                  output_hidden_states=True).hidden_states[-1]
            # [B, L+2, H]
            return cross_block_hidden_states[:, :-2, :]

        else:
            if self.config.reuse_transformer_for_cross_block_trans:
                cross_block_hidden_states = trans_cache_hidden_states
            else:
                cross_block_hidden_states = self.cross_block_trans_base_model.forward(input_ids = argv["input_ids"], labels = argv.get("labels"), \
                            attention_mask = argv.get("attention_mask"), output_hidden_states=True).hidden_states[-1]
                    
        return cross_block_hidden_states

    def forward(self, input_time, **argv):
        return_rates_matrix = argv.get("return_rates_matrix", False)
        return_init_prob = argv.get("return_init_prob", False)
        return_cross_block_trans = argv.get("return_cross_block_trans", False)

        trans_cache_hidden_states = self.base_models["trans_base"].forward(input_ids = argv["input_ids"], labels = argv.get("labels"), \
                        attention_mask = argv.get("attention_mask"), output_hidden_states=True).hidden_states[-1]
        
        if self.config.transformer_offset:
            offsets_cache_hidden_states = self.base_models["offsets_base"].forward(input_ids = argv["input_ids"], labels = argv.get("labels"), \
                            attention_mask = argv.get("attention_mask"), output_hidden_states=True).hidden_states[-1]
        else:
            offsets_cache_hidden_states = trans_cache_hidden_states
        
        continents = argv.get(self.config.data_properties[0])
        countries = argv.get(self.config.data_properties[1])

        all_country_logtis = []
        all_sum_of_country_logits = []
        all_country_trans_rates = [] # transition rate matrices for countries
        all_country_init_probs = [] # initial probability for countries

        indices = []

        for i in range(len(self.num_components)):
            mask = (continents == i)
            if torch.sum(mask).item() <= 0:
                continue
            indices.append(torch.nonzero(mask))
            sub_input_time, sub_argv = self._split_batch(mask, input_time, trans_cache_hidden_states=trans_cache_hidden_states, \
                offsets_cache_hidden_states=offsets_cache_hidden_states,
                **argv)
            country_logits = self.local_models[i].forward(sub_input_time, **sub_argv, return_component_logits=True) # [B, L, V, K]  return_logits = True
            if return_rates_matrix:
                all_country_trans_rates.extend(list(country_logits.info_dict["rates_matrix"]))
            if return_init_prob:
                all_country_init_probs.extend(list(country_logits.info_dict["init_prob"]))

            all_sum_of_country_logits.append(
                torch.logsumexp(country_logits.info_dict["component_logits"], dim=-1) 
                - (math.log(country_logits.info_dict["component_logits"].size(-1)) if getattr(self.config, "apply_log_softmax", False) else 0.0)
                )
            all_country_logtis.append(country_logits.logits)

        all_country_logtis = torch.cat(all_country_logtis, dim=0)
        all_sum_of_country_logits = torch.cat(all_sum_of_country_logits, dim=0)
        indices = torch.cat(indices, dim=0).squeeze(-1)
        country_logtis = all_country_logtis[torch.argsort(indices)] # [B, L, V]
        sum_of_country_logits = all_sum_of_country_logits[torch.argsort(indices)] # Could be used to train global model
        
        # collect information from countries
        if len(all_country_trans_rates) > 0:
            all_country_trans_rates = [all_country_trans_rates[x.item()] for x in torch.argsort(indices)]
            all_country_trans_rates = [torch.gather(x, 1, argv["input_ids"][i].view(argv["input_ids"].size(1), 1, 1, 1).expand(-1, -1, x.size(-2), x.size(-1))).squeeze(1) for i, x in enumerate(all_country_trans_rates)]
        if len(all_country_init_probs) > 0:
            all_country_init_probs = [all_country_init_probs[x.item()] for x in torch.argsort(indices)]
            all_country_init_probs = [torch.gather(x, 1, argv["input_ids"][i].view(argv["input_ids"].size(1), 1, 1).expand(-1, -1, x.size(-1))).squeeze(1) for i, x in enumerate(all_country_init_probs)]
            
        ## 2. Get continents outputs
        continent_outputs = self.get_continent_evolution_outputs(
            input_time, 
            trans_cache_hidden_states=trans_cache_hidden_states, 
            offsets_cache_hidden_states=offsets_cache_hidden_states, **argv)
        continent_logits = continent_outputs.logits

        ## 3. Calculate transmission from continents to coutries
        cross_block_hidden_states = self.cross_block_trans_base_model_forward(argv, trans_cache_hidden_states)
        if self.config.block_trans_model == "continent_to_continent":
            cross_continent_logits, cross_block_trans = self.block_trans_continent_to_continent(
                cross_block_hidden_states, continents, countries, 
                continent_logits=torch.exp(continent_outputs.info_dict["component_logits"]),
                return_trans_weight=return_cross_block_trans
                )
        elif self.config.block_trans_model == "country_to_continent":
            cross_continent_logits, cross_block_trans  = self.block_trans_country_to_continent(
                cross_block_hidden_states, continents, countries, 
                continent_logits=torch.exp(continent_outputs.info_dict["component_logits"]),
                return_trans_weight=return_cross_block_trans)
        
        total_logits = torch.log(torch.exp(country_logtis) + cross_continent_logits + self.eps) 
        
        info_dict = {
            "continent_logits": continent_logits, 
            "sum_of_country_logits": sum_of_country_logits, 
            "cross_group_trans_weight": cross_block_trans["cross_group_trans_weight"]}
        
        if return_cross_block_trans:
            for key in info_dict:
                ssize = list(info_dict[key].size())
                ssize[2] = 1
                input_ids_expand = argv["input_ids"].view(argv["input_ids"].size() + (1,) * (info_dict[key].dim() - 2)) # [B, L]
                input_ids_expand = input_ids_expand.expand(*ssize)
                info_dict[key] = torch.gather(info_dict[key], 2, input_ids_expand).squeeze(2)
        
        if return_rates_matrix:
            info_dict["country_trans_rates"] = all_country_trans_rates
        if return_init_prob:
            info_dict["country_init_probs"] = all_country_init_probs

        return GPTOutputs(logits=total_logits, info_dict=info_dict)

@register_model("gpt2_time_transmission_hierarchy")
class GPT2TimeTransmissionHierarchy(GPT2TimeTransmission):
    def __init__(self, config, alphabet) -> None:
        super().__init__(config, alphabet)

    def initialize_model(self, pretrained_model_name_or_path: str):
        """create and initialize the model to use with this task,
        Feel free to overwrite this method if you are initializing the model in a different way
        """
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path="gpt2", **self.model_data_kwargs
        )

        setattr(config, "offset_pos_function", getattr(self.config, "offset_pos_function", "softmax"))
        setattr(config, "pos_function", getattr(self.config, "pos_function", "softplus"))
        setattr(config, "trans_w_pos_function", getattr(self.config, "trans_w_pos_function", "softplus"))
        setattr(config, "max_rate_value", getattr(self.config, "max_rate_value", 1e5))
        setattr(config, "min_rate_value", getattr(self.config, "min_rate_value", 1e-12))
        setattr(config, "normalize_time_a", self.config.normalize_time_a)
        setattr(config, "normalize_time_b", self.config.normalize_time_b)
        setattr(config, "transformer_offset", self.config.transformer_offset)
        setattr(config, "data_properties", self.config.data_properties)
        setattr(config, "output_layer_type", getattr(self.config, "output_layer_type", "linear"))
        setattr(config, "block_trans_model", self.config.block_trans_model)
        setattr(config, "block_trans_model_n_layer", 
                getattr(self.config, "block_trans_model_n_layer", self.config.num_hidden_layers))
        setattr(config, "implement_version", self.config.implement_version)        
        setattr(config, "reuse_transformer_for_cross_block_trans", \
            getattr(self.config, "reuse_transformer_for_cross_block_trans", False))
        setattr(config, "use_simple_continent_model", \
            getattr(self.config, "use_simple_continent_model", False))
        setattr(config, "continent_share_base_models", \
            getattr(self.config, "continent_share_base_models", True))
        setattr(config, "contient2country", self.config.contient2country) 
        setattr(config, "use_linear_init_trans_weight", \
                getattr(self.config, "use_linear_init_trans_weight", False)) 
        output_layer_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path="gpt2", 
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            num_hidden_layers=getattr(self.config, "output_layer_num_hidden_layers", self.config.num_hidden_layers)
        )
        number_of_conponents = [len(x[1]) for x in self.config.contient2country]
        self.model = GPT2TimeTransmissionHierarchyModule.from_config(config, number_of_conponents, output_layer_config=output_layer_config)

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
        model = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict=False) # TODO: for debug...
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
        parent_parser = super(GPT2TimeTransmissionHierarchy, cls).add_argparse_args(parent_parser)
        parent_parser.add_argument('--block_trans_model', type=str, default="country_to_continent", 
                                   choices=["country_to_continent", "continent_to_continent"])
        parent_parser.add_argument('--continent_loss_weight', type=float, default=0)
        parent_parser.add_argument('--reuse_transformer_for_cross_block_trans', type=str2bool, default='true')
        parent_parser.add_argument('--use_simple_continent_model', type=str2bool, default='true')
        parent_parser.add_argument('--continent_share_base_models', type=str2bool, default='true')
        parent_parser.add_argument('--cross_continent_reg', type=float, default=0.0)
        parent_parser.add_argument('--output_info', type=str2bool, default='false', help="Output info like rate matrix in testing time.")
        parent_parser.add_argument('--trans_w_pos_function', type=str, default="softplus", 
                                   choices=["softplus", "sigmoid", "relu", "none"])
        parent_parser.add_argument('--block_trans_model_n_layer', type=int, default=4)
        parent_parser.add_argument('--use_linear_init_trans_weight', type=str2bool, default='false')
        return parent_parser
        
    def calc_continent_loss(self, continent_logits, labels, loss_weight, reduce):
        B = continent_logits.size(0)
        assert continent_logits.size() == labels.size()
        loss = torch.mean((continent_logits.view(B, -1) - labels.view(B, -1)) ** 2, dim=-1) # [B]
        if reduce:
            if loss_weight is not None:
                loss_weight = loss_weight / loss_weight.sum()
                loss = torch.sum(loss * loss_weight)
            else:
                loss = loss.mean()
        return loss

    def forward(self, batch, batch_idx, reduce=True, mode="train"):
        if getattr(self.config, "zero_time", False):
            batch["input_time"].fill_(0.)

        if getattr(self.config, "set_time", None) is not None:
            batch["input_time"].fill_(self.config.set_time) # set time bin as a constant.

        model_outputs = self.model(**batch, 
                                   return_rates_matrix = self.config.output_info, 
                                   return_init_prob = self.config.output_info,
                                   return_cross_block_trans = self.config.output_info)
        
        if self.config.weight_loss_by_count and batch.get('freq', None) is not None and batch.get('bin_size', None) is not None:
            loss_weight = batch.get('freq', None) * batch.get('bin_size', None) # TODO change from batch.get('freq', None)
        else:
            loss_weight = 1.0
            
        labels = batch["labels"]
        country_loss = self.nll_loss(model_outputs.logits, labels, loss_weight=loss_weight, 
                                     reduce=reduce, ignore_bos=True if mode == "test" else False)
        loss_dict = {"country_loss": country_loss}

        if mode != "test" and self.config.continent_loss_weight > 0 and "continent_logits" in model_outputs.info_dict:
            continent_loss = self.calc_continent_loss(
                model_outputs.info_dict["continent_logits"], 
                model_outputs.info_dict["sum_of_country_logits"], 
                loss_weight, reduce)
            loss_dict["continent_loss"] = continent_loss
            loss = country_loss + self.config.continent_loss_weight * continent_loss
        else:
            loss = country_loss
        
        if  mode != "test" and getattr(self.config, "cross_continent_reg", 0.0) > 0.0 and "cross_group_trans_weight" in model_outputs.info_dict:
            cross_group_trans_weight = model_outputs.info_dict["cross_group_trans_weight"]
            cross_group_l2_loss = torch.mean(cross_group_trans_weight ** 2)
            loss_dict["cross_continent_reg_loss"] = cross_group_l2_loss
            loss = loss + self.config.cross_continent_reg * cross_group_l2_loss

        if self.config.output_info: # collect information for analysis
            for key in model_outputs.info_dict:
                loss_dict[key] = model_outputs.info_dict[key]
            
        return loss, loss_dict # model_outputs.info_dict

    def overwrite_generate_kwargs(self, new_config):
        super().overwrite_generate_kwargs(new_config)
        setattr(self.config, "output_info", new_config.output_info)