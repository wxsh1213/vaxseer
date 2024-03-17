import os, torch, sys, logging
from config import parse_args
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar, EarlyStopping
from models import build_model
from data.data_modules import build_data_module
import csv, json
import pickle as pkl

def save_results(predict_dataset, predictions, save_path=None, clean_output=True):
    # acc_id, sequence, time, weight, predicted_weight
    logging.info("Save predictions to %s" % save_path)
    with open(save_path, 'w') as csvfile:
        spamwriter = csv.writer(csvfile)
        dataset_keys = list(predict_dataset[0].keys())
        # print(dataset_keys)
        pred_keys = list(predictions[0].keys())
        if clean_output:
            pred_keys = [x for x in pred_keys if x != "index"]
            dataset_keys = [x for x in dataset_keys if x in ("src_id", "src_time", "freq") and x not in pred_keys]
            

        keys = dataset_keys + pred_keys
        logging.info("Saving keys: %s" % " ".join(keys))
        spamwriter.writerow(keys)
        for i in range(len(predictions)):
            dataset_index = predictions[i].get("index", i)
            # print(dataset_index, i, len(predict_dataset), len(predictions))
            row = [predict_dataset[dataset_index][k] for k in dataset_keys] + [predictions[i][k] for k in pred_keys]
            spamwriter.writerow(row)
    
    if "prediction" in predictions[0] and isinstance(predictions[0]["prediction"], str): # Output sequences
        save_fasta_file = save_path.split(".csv")[0] + ".fasta"
        logging.info("Save sequences to %s" % save_fasta_file)
        with open(save_fasta_file, "w") as fout:
            for i in range(len(predictions)):
                dataset_index = predictions[i].get("index", i)
                row = ["%s=%s" % (k, str(predict_dataset[dataset_index][k])) for k in dataset_keys] + ["%s=%s" % (k, str(predictions[i][k])) for k in pred_keys if k != "prediction"]
                desc = "|".join(row)
                fout.write(">%s\n%s\n\n" % (desc, predictions[dataset_index]["prediction"]))
            
def overwrite_generate_kwargs(model, new_config):
    if hasattr(model, "config"):
        config = model.config
    if hasattr(model, "args"):
        config = model.args

    setattr(config, "generate_temperature", getattr(new_config, "generate_temperature", 1.0))
    if getattr(new_config, "generate_max_length", None) is None:
        setattr(config, "generate_max_length", config.max_position_embeddings)
    else:
        setattr(config, "generate_max_length", new_config.generate_max_length) #  = getattr(self.config, "generate_max_length", None)
    
    setattr(config, "generate_do_sample", getattr(new_config, "generate_do_sample", True))

    if hasattr(model, "overwrite_generate_kwargs"):
        model.overwrite_generate_kwargs(new_config)


def train(args, model, dm):
    root_dir = args.default_root_dir # os.path.join(args., "ReverseTask")
    os.makedirs(root_dir, exist_ok=True)
    # args.early_step
    callbacks = [
            ModelCheckpoint(save_weights_only=False, mode=args.model_ckpt_mode, monitor=args.model_ckpt_monitor, save_last=True, every_n_train_steps=args.model_ckpt_every_n_train_steps),
            LearningRateMonitor("step"),
            TQDMProgressBar(refresh_rate=1),
            # EarlyStopping(monitor=args.early_stop_monitor, mode=args.early_stop_mode, patience=args.early_stop_patience), # Early stop is based on the eval steps
            # SaveConfigCallBacks
        ]
    if args.early_stop:
        callbacks.append(EarlyStopping(monitor=args.early_stop_monitor, mode=args.early_stop_mode, patience=args.early_stop_patience))
    
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    if args.predict:
        overwrite_generate_kwargs(model, args)
        predictions = trainer.predict(model, dataloaders=dm.predict_dataloader()) # TODO
        predictions = [ele for batch in predictions for ele in batch]
        # print(len(predictions), predictions[0])
        predict_dataset = [x for pred_set in dm.pred_datasets for x in pred_set]
        output_path = args.save_prediction_path
        if output_path is None:
            output_path = os.path.join(trainer.logger.log_dir, "predictions.csv")
        
        results = model.output_predicting_results(predictions, predict_dataset, output_path)

        logging.info("Save testing results to %s" % output_path)
        if results:
            with open(output_path, 'w') as csvfile:
                spamwriter = csv.writer(csvfile)
                keys = list(results[0].keys())
                spamwriter.writerow(keys)
                for i in range(len(results)):
                    spamwriter.writerow([results[i][k] for k in keys])

    elif args.test:
        overwrite_generate_kwargs(model, args)
        trainer.test(model, dataloaders=dm.test_dataloader())
        
        if hasattr(model, "all_outputs"):  # saving the testing results
            outputs = model.all_outputs
            results = model.output_testing_results(outputs, dm.test_datasets)
            
            output_path = args.save_prediction_path
            if output_path is None:
                output_path = os.path.join(trainer.logger.log_dir, "test_results.csv")
            
            logging.info("Save testing results to %s" % output_path)
            with open(output_path, 'w') as csvfile:
                spamwriter = csv.writer(csvfile)
                keys = list(results[0].keys())
                spamwriter.writerow(keys)
                for i in range(len(results)):
                    spamwriter.writerow([results[i][k] for k in keys])
    else:
        trainer.fit(model, datamodule=dm, ckpt_path=args.resume_from_checkpoint)

def get_testing_time(args):
    max_testing_time, min_testing_time = -1, -1    
    if args.max_testing_time != -1:
        max_testing_time = args.max_testing_time
    if args.min_testing_time != -1:
        min_testing_time = args.min_testing_time
    
    return max_testing_time, min_testing_time

if __name__ == "__main__":
    args = parse_args() # , model_cls=GPT2Time # dm_cls=ProteinLMWeightedDataModule
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    pl.seed_everything(args.seed, workers=True)    
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    # torch.autograd.set_detect_anomaly(True)

    torch.multiprocessing.set_sharing_strategy('file_system') # in case opening too many files error

    if args.test or args.predict:
        max_testing_time, min_testing_time = get_testing_time(args)
        model = build_model(args.model).load_from_checkpoint(args.resume_from_checkpoint, \
            max_testing_time=max_testing_time, \
            min_testing_time=min_testing_time,  # model_name_or_path=args.model_name_or_path, 
            args=args)
        setattr(model, "max_testing_time", max_testing_time)
        setattr(model, "min_testing_time", min_testing_time)

        data_module = build_data_module(args.data_module)(args, vocab=model.alphabet) # use the alphabet from the checkpoints
        data_module.prepare_data()
            
        if hasattr(model, "config"):
            data_module.setup(stage="test" if args.test else "predict", model_config=model.config)
        elif hasattr(model, "args"):
            data_module.setup(stage="test" if args.test else "predict", model_config=model.args)
        else:
            raise ValueError("Unknown model args.")
    else:
        print(args)
        data_module = build_data_module(args.data_module)(args) #  ProteinLMWeightedDataModule(args)
        data_module.prepare_data()
        data_module.setup(stage="test" if args.test else "fit")
        if getattr(args, "load_from_checkpoint", None):
            model = build_model(args.model).load_from_checkpoint(checkpoint_path=args.load_from_checkpoint, args=args)
        else:
            model = build_model(args.model)(args, alphabet=data_module.vocab)
    
    result = train(args, model, data_module)