import pandas as pd
import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import argparse
from sklearn.metrics import average_precision_score, roc_auc_score
from Bio import SeqIO

torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--training_path', default="", type=str)
    parser.add_argument('--valid_path', default="", type=str)
    parser.add_argument('--test_path', default="", type=str)
    parser.add_argument('--ref_seq_path', default="", type=str)
    parser.add_argument('--ckpt_saving_dir', default="", type=str)
    parser.add_argument('--testing_result_saving_path', default="", type=str)
    parser.add_argument('--model', default="linear_regression", type=str)

    parser.add_argument('--valid_vaccine_ref_aln_path', default="", type=str)
    parser.add_argument('--valid_virus_ref_aln_path', default="", type=str)
    parser.add_argument('--train_vaccine_ref_aln_path', default="", type=str)
    parser.add_argument('--train_virus_ref_aln_path', default="", type=str)
    parser.add_argument('--test_vaccine_ref_aln_path', default="", type=str)
    parser.add_argument('--test_virus_ref_aln_path', default="", type=str)
    
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    
    args = parser.parse_args()
    return args


def read_ref_alignment(path):
    alingments = dict()
    with open(path) as fin:
        for line in fin:
            query,target,qaln,taln,qstart,qend,tstart,tend,mismatch = line.strip().split()
            alingments[query] = (taln, qaln, int(tstart))
    return alingments

def get_aa_subs_from_alignment(ref_seq, aln_seq, ref_start):
    new_ref_aa = []
    new_aln_aa = []
    for i, (ref_aa, aln_aa) in enumerate(zip(ref_seq, aln_seq)):
        if ref_aa != "-":
            new_ref_aa.append(ref_aa)
            new_aln_aa.append(aln_aa)
    
    aa_subs = []
    for i, (ref_aa, aln_aa) in enumerate(zip(new_ref_aa, new_aln_aa)):
        if ref_aa != aln_aa and aln_aa != "-":
            aa_subs.append((ref_aa, i + ref_start - 1, aln_aa))
    return aa_subs

def build_dataset(df, virus2aa_subs, add_virus_indicator=False, add_vaccine_indicator=False):
    inputs = []
    targets = []
    

    for virus_id, ref_id, label in zip(df["virus"], df["reference"], df["hi"]):
        virus_aa_subs = virus2aa_subs[virus_id]
        vaccine_aa_subs = vaccine2aa_subs[ref_id]
        new_virus_aa_subs = {x[1]: x for x in virus_aa_subs if x not in vaccine_aa_subs}
        new_vaccine_aa_subs = {x[1]: x for x in vaccine_aa_subs if x not in virus_aa_subs}
        new_input = []
        for pos in set(new_virus_aa_subs.keys()) | set(new_vaccine_aa_subs.keys()):
            if pos in new_vaccine_aa_subs:
                vaccine_type = new_vaccine_aa_subs[pos][-1]
            else:
                vaccine_type = ref_seq[pos]
            if pos in new_virus_aa_subs:
                virus_type = new_virus_aa_subs[pos][-1]
            else:
                virus_type = ref_seq[pos]
            new_input.append("%s%d%s" % (vaccine_type, pos, virus_type))
            new_input.append("%s%d%s" % (virus_type, pos, vaccine_type))

        inputs.append((virus_id, ref_id, new_input))
        targets.append(label)
    
    return inputs, targets

def vectorize(inputs, targets, feature_dict, virus_dict, vaccine_dict):
    inputs_tensor = torch.zeros(size=(len(inputs), len(feature_dict)))
    labels_tensor = torch.zeros(size=(len(inputs), 1))

    virus_indicator = torch.zeros(size=(len(inputs), len(virus_dict)))
    vaccine_indicator = torch.zeros(size=(len(inputs), len(vaccine_dict)))

    for i in range(len(inputs)):
        virus_id, ref_id, aa_subs = inputs[i]
        if virus_id in virus_dict:
            virus_indicator[i, virus_dict[virus_id]] = 1.0
        if ref_id in vaccine_dict:
            vaccine_indicator[i, vaccine_dict[ref_id]] = 1.0

        for aa in aa_subs:
            if aa not in feature_dict:
                continue
            inputs_tensor[i, feature_dict[aa]] = 1.0
        labels_tensor[i, 0] = targets[i]


    

    return inputs_tensor, virus_indicator, vaccine_indicator, labels_tensor


class NeherModel(nn.Module):
    def __init__(self, num_feats, ouput_dim, num_vaccines, num_virus) -> None:
        super().__init__()
        self.params = nn.Parameter(torch.randn(num_feats, ouput_dim))
        self.vaccine_bias = nn.Parameter(torch.randn(num_vaccines, 1))
        self.virus_bias = nn.Parameter(torch.randn(num_virus, 1))
        self.relu = nn.ReLU()

    def forward(self, input_tensor, virus_indicator, vaccine_indicator):
        # input_tensor: [B, F], vaccine_indicator: [B], virus_indicator: [B]
        logits = input_tensor @ self.relu(self.params) # [B, 1]
        logits += vaccine_indicator @ self.vaccine_bias  # [B, 1]
        logits += virus_indicator @ self.virus_bias # [B, 1]
        return logits


class NeherLoss(nn.Module):
    def __init__(self, l, g, d, model) -> None:
        super().__init__()
        self._lambda = l
        self._gamma = g
        self._delta = d
        self._model = model
        self.relu = nn.ReLU()
    
    def forward(self, pred, target):
        loss = torch.mean((pred - target) ** 2)
        loss += self._lambda * torch.mean(self.relu(self._model.params))
        loss += self._gamma * torch.mean(self._model.virus_bias ** 2)
        loss += self._delta * torch.mean(self._model.vaccine_bias ** 2)
        return loss

class LinearRegression(nn.Module):
    def __init__(self, num_feats, ouput_dim) -> None:
        super().__init__()
        self.params = nn.Parameter(torch.randn(num_feats, ouput_dim))
    
    def forward(self, input_tensor, **args):
        # input_tensor: [B, F]
        logits = input_tensor @ self.params
        # mse_loss = torch.mean((logits - target_tensor) ** 2)
        return logits


def test(dataloader, model, loss_fn, print_all=False):
    model.eval()

    test_loss = 0.0
    total_sample = 0.0
    all_ys = []
    all_preds = []
    with torch.no_grad():
        for X, virus, vaccine, y in dataloader:
            X, y, virus, vaccine = X.to(device), y.to(device), virus.to(device), vaccine.to(device)
            pred = model(X, virus, vaccine)
            test_loss += loss_fn(pred, y).item() * X.size(0)
            total_sample += X.size(0)
            
            all_ys.append(y.cpu())
            all_preds.append(pred.cpu())


    test_loss = test_loss / total_sample

    all_ys = torch.cat(all_ys)
    all_preds = torch.cat(all_preds)

    try:
        prauc = average_precision_score((all_ys>=3).long(), all_preds)
        roc_auc = roc_auc_score((all_ys>=3).long(), all_preds)
    except Exception as e:
        print(e)
        prauc, roc_auc = 0.0, 0.0

    mse = torch.mean((all_ys - all_preds) ** 2)

    if print_all:
        return all_preds, {"mse": mse.item(), "roc_auc": roc_auc, "pr_auc": prauc}
    else:
        return {"mse": mse.item(), "roc_auc": roc_auc, "pr_auc": prauc}


def train(train_dataloader, valid_dataloader, model, loss_fn, optimizer, epoch_num=100, ckpt_dir=None, verbose=True):
    step = 0

    performances = []
    
    for epoch in range(epoch_num):
        model.train()
        # print("Epoch", epoch)
        for batch, (X, virus, vaccine, y) in enumerate(train_dataloader):
            step += 1

            X, y, virus, vaccine = X.to(device), y.to(device), virus.to(device), vaccine.to(device)
            pred = model(X, virus, vaccine)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # manage the best
        valid_loss = test(valid_dataloader, model, loss_fn)["mse"]
        # if step % 100 == 0:
        if verbose:
            print("Epoch %d, train loss: %g, valid_loss (mse): %g" % (epoch, loss.item(), valid_loss))
        
        if len(performances) == 0 or valid_loss < min(performances):
            if verbose:
                print("Saving checkpoint to %s" % (os.path.join(ckpt_dir, "best.ckpt")))
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.ckpt"))
        performances.append(valid_loss)



if __name__ == "__main__":

    args = parse_args()

    train_df = pd.read_csv(args.training_path)
    valid_df = pd.read_csv(args.valid_path)
    test_df = pd.read_csv(args.test_path)

    train_vaccine_aln = read_ref_alignment(args.train_vaccine_ref_aln_path)
    train_virus_aln = read_ref_alignment(args.train_virus_ref_aln_path)

    valid_vaccine_aln = read_ref_alignment(args.valid_vaccine_ref_aln_path)
    valid_virus_aln = read_ref_alignment(args.valid_virus_ref_aln_path)

    test_vaccine_aln = read_ref_alignment(args.test_vaccine_ref_aln_path)
    test_virus_aln = read_ref_alignment(args.test_virus_ref_aln_path)

    ref_seq = SeqIO.parse(args.ref_seq_path, "fasta")
    ref_seq = str(next(ref_seq).seq)
    print(ref_seq)

    vaccine2aa_subs = {}
    train_vaccine_set = set()
    for vaccine_id in train_vaccine_aln:
        aa_subs = get_aa_subs_from_alignment(*train_vaccine_aln[vaccine_id])
        vaccine2aa_subs[vaccine_id] = aa_subs
        train_vaccine_set.add(vaccine_id)
    for vaccine_id in valid_vaccine_aln:
        aa_subs = get_aa_subs_from_alignment(*valid_vaccine_aln[vaccine_id])
        vaccine2aa_subs[vaccine_id] = aa_subs
    for vaccine_id in test_vaccine_aln:
        aa_subs = get_aa_subs_from_alignment(*test_vaccine_aln[vaccine_id])
        vaccine2aa_subs[vaccine_id] = aa_subs
        
    virus2aa_subs = {}
    train_virus_set = set()
    for virus_id in train_virus_aln:
        aa_subs = get_aa_subs_from_alignment(*train_virus_aln[virus_id])
        virus2aa_subs[virus_id] = aa_subs
        train_virus_set.add(virus_id)
    for virus_id in valid_virus_aln:
        aa_subs = get_aa_subs_from_alignment(*valid_virus_aln[virus_id])
        virus2aa_subs[virus_id] = aa_subs
    for virus_id in test_virus_aln:
        aa_subs = get_aa_subs_from_alignment(*test_virus_aln[virus_id])
        virus2aa_subs[virus_id] = aa_subs

    training_inputs, training_targets = build_dataset(train_df, virus2aa_subs)
    # print(len(training_inputs), training_inputs[0])
    valid_inputs, valid_targets = build_dataset(valid_df, virus2aa_subs)
    test_inputs, test_targets = build_dataset(test_df, virus2aa_subs)
    
    feature_list = list(set([x for y in training_inputs for x in y[-1]]))
    feature_list.sort()
    feature_dict = {x: i for i, x in enumerate(feature_list)}


    virus_list = list(train_virus_set)
    virus_list.sort()
    virus_dict = {x: i for i, x in enumerate(virus_list)}
    vaccine_list = list(train_vaccine_set)
    vaccine_list.sort()
    vaccine_dict = {x: i for i, x in enumerate(vaccine_list)}

    # inputs_tensor, virus_indicator, vaccine_indicator, labels_tensor
    training_inputs_tensor,  training_virus_indicator,  training_vaccine_indicator, training_targets_tensor = vectorize(training_inputs, training_targets, feature_dict, virus_dict, vaccine_dict)
    valid_inputs_tensor, valid_virus_indicator,  valid_vaccine_indicator, valid_targets_tensor = vectorize(valid_inputs, valid_targets, feature_dict, virus_dict, vaccine_dict)
    test_inputs_tensor, test_virus_indicator,  test_vaccine_indicator, test_targets_tensor = vectorize(test_inputs, test_targets, feature_dict, virus_dict, vaccine_dict)
    print("Training set size:", training_inputs_tensor.size(), training_targets_tensor.size())
    print("Valid set size:", valid_inputs_tensor.size(), valid_targets_tensor.size())
    print("Test set size:", test_inputs_tensor.size(), test_targets_tensor.size())
    # print(training_virus_indicator.size(), training_vaccine_indicator.size())
    # exit()


    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    if args.model == "linear_regression":
        model = LinearRegression(training_inputs_tensor.size(1), 1).to(device)
        loss_fn = torch.nn.MSELoss()
    elif args.model == "neher":
        # num_feats, ouput_dim, num_vaccines, num_virus
        model = NeherModel(training_inputs_tensor.size(1), 1, len(vaccine_dict), len(virus_dict)).to(device)
        loss_fn = NeherLoss(l=1.0, g=2.0, d=0.2, model=model)

    print(model)
    batch_size = 128

    if not args.test:
        train_dataloader = DataLoader(TensorDataset(training_inputs_tensor, training_virus_indicator,  training_vaccine_indicator, training_targets_tensor), batch_size=batch_size)
        valid_dataloader = DataLoader(TensorDataset(valid_inputs_tensor, valid_virus_indicator,  valid_vaccine_indicator, valid_targets_tensor), batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(train_dataloader, valid_dataloader, model, loss_fn, optimizer, ckpt_dir=args.ckpt_saving_dir, verbose=args.verbose)

    # Testing for the best model
    test_dataloader = DataLoader(TensorDataset(test_inputs_tensor, test_virus_indicator,  test_vaccine_indicator, test_targets_tensor), batch_size=batch_size)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_saving_dir, "best.ckpt")))
    all_preds, test_loss = test(test_dataloader, model, loss_fn, print_all=True)
    print("================= Testing Results =================")
    for key in test_loss:
        print(key, test_loss[key])
    
    # saving testing results
    test_df["label"] = all_preds.numpy()
    if not os.path.exists(os.path.split(args.testing_result_saving_path)[0]):
        os.makedirs(os.path.split(args.testing_result_saving_path)[0])
    print("Saving results to %s" % (args.testing_result_saving_path))
    test_df.to_csv(args.testing_result_saving_path)

    # Validation for the best model
    valid_dataloader = DataLoader(TensorDataset(valid_inputs_tensor, valid_virus_indicator,  valid_vaccine_indicator, valid_targets_tensor), batch_size=batch_size)
    all_preds, test_loss = test(valid_dataloader, model, loss_fn, print_all=True)
    print("================= Validation Results =================")
    for key in test_loss:
        print(key, test_loss[key])






    
