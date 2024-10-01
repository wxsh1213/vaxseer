import logging, argparse, os
import pytorch_lightning as pl
from models import build_model
from data.data_modules import build_data_module
from utils.args import str2bool

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def parse_args(dm_cls=None, model_cls=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=1005)
    
    parser.add_argument('--early_stop', action="store_true")
    parser.add_argument('--early_stop_patience', type=int, default=3)
    parser.add_argument('--early_stop_monitor', type=str, default="val_loss")
    parser.add_argument('--early_stop_mode', type=str, default="min")
    parser.add_argument('--model_ckpt_monitor', type=str, default="val_loss")
    parser.add_argument('--model_ckpt_mode', type=str, default="min")
    parser.add_argument('--model_ckpt_every_n_train_steps', type=int, default=None, help="Save the model checkpoints every N training batches. Should run the validation steps first if we want to save according to val loss.")
    parser.add_argument('--test', action="store_true", help="Testing mode.")
    parser.add_argument('--predict', action="store_true", help="Prediction mode.")
    parser.add_argument('--save_prediction_path', type=str, default=None)
    parser.add_argument('--cudnn_deterministic', type=str2bool, default="true", help="Prediction mode.")
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('-d', '--data_module', type=str, default="lm_weighted")
    parser.add_argument('--max_testing_time', type=int, default=-1)
    parser.add_argument('--min_testing_time', type=int, default=-1)
    parser.add_argument('--weight_decay_rate', type=float, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--scheduler', type=str, default="linear", choices=["cosine", "linear", "none"])

    # add the args from Trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # add the args for DataModule
    if dm_cls is not None:
        parser = dm_cls.add_argparse_args(parser)
    else:
        parsed, unparsed = parser.parse_known_args()
        parser = build_data_module(parsed.data_module).add_argparse_args(parser)

    # add the args for Model
    if model_cls is not None:
        parser = model_cls.add_argparse_args(parser)
    else:
        parsed, unparsed = parser.parse_known_args()
        parser = build_model(parsed.model).add_argparse_args(parser)
    
    # Parsing
    args = parser.parse_args()
    
    # If the vocab type (e.g., ESM, MSA etc...) is not specified, we use the vocab type from model_name_or_path.
    if args.vocab == "" and args.model_name_or_path != "":
        if "esm_msa" in args.model_name_or_path:
            args.vocab = "msa"
        else:
            args.vocab = os.path.split(args.model_name_or_path)[-1].split("_")[0]

    return args


