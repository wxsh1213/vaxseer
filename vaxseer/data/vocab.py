from warnings import WarningMessage
from esm import Alphabet
import logging

def load_esm_alphabet(vocab_version, mol_type="protein"):
    # return Alphabet.from_architecture(vocab_version)
    # vocab_version = os.path.split(cfg.model_name_or_path)[-1].split("_")[0]
    if vocab_version == "esm1":
        alphabet = Alphabet.from_architecture("ESM-1", mol_type=mol_type)
    elif vocab_version == "esm1b" or vocab_version == "esm1v": # both are roberta_large:
        alphabet = Alphabet.from_architecture("roberta_large", mol_type=mol_type)
    elif vocab_version == "msa":
        alphabet = Alphabet.from_architecture("msa_transformer", mol_type=mol_type)
    else:
        alphabet = Alphabet.from_architecture(vocab_version, mol_type=mol_type)
        # logging.warn("Unknown vocab type: %s, use roberta_large vocab by default." % vocab_version)
        
    # alphabet = Alphabet.from_architecture(cfg.vocab_version)
    return alphabet