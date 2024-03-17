from Bio import SeqIO
from collections import defaultdict
import math, os, subprocess, argparse
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

torch.manual_seed(0)

def build_dataset(fasta_path, alignment_path, ref_seq_path):
    mmseqs_alignment(alignment_path, fasta_path, ref_seq_path)
    alignment = read_ref_alignment(alignment_path)
    accid2aa_subs = {}
    for accid in alignment:
        aa_subs = get_aa_subs_from_alignment(*alignment[accid])
        accid2aa_subs[accid] = aa_subs

    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        
        freq = float({x.split("=")[0]: x.split("=")[1] for x in record.description.split()[1].split("|")}["freq"])
        time_bin = int({x.split("=")[0]: x.split("=")[1] for x in record.description.split()[1].split("|")}["time_bin"])
        bin_size = int({x.split("=")[0]: x.split("=")[1] for x in record.description.split()[1].split("|")}["bin_size"])
        
    
        records.append({
            "id": record.id, 
            "seq": apply_aa_subs(ref_seq, accid2aa_subs[record.id]), 
            "freq": freq, 
            "time_bin": time_bin, 
            "bin_size": bin_size,
            })

    return records # , temporal_aa2freq, seq_len

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

def build_freq_matrix(times, temporal_aa2freq, vocab, seq_len):
    input_tensor = torch.zeros(len(times), seq_len, len(vocab)) # [T, V]
    for i, t in enumerate(temporal_aa2freq):
        for pos in temporal_aa2freq[t]:
                for aa in temporal_aa2freq[t][pos]:
                    freq = temporal_aa2freq[t][pos][aa]
                    input_tensor[i, pos, vocab.get(aa, len(vocab)-1)] = freq
    return input_tensor, torch.tensor(times)

def apply_aa_subs(complete_ref_seq, aa_subs):
    ref_aas = list(complete_ref_seq)
    for wt, loc, mut in aa_subs:
        assert complete_ref_seq[loc] == wt
        ref_aas[loc] = mut
    return "".join(ref_aas)


def mmseqs_alignment(alignment_path, query_path, ref_seq_path):
    # query_path="/data/rsg/nlp/wenxian/esm/data/who_flu/before_2018-04/a_h3n2_virus.fasta"
    # target_path="/data/rsg/nlp/wenxian/esm/data/gisaid/flu/reference_fasta/prortein_A_NewYork_392_2004_H3N2_ha.fasta"
    # save_path="/data/rsg/nlp/wenxian/esm/data/who_flu/before_2018-04/a_h3n2_virus.ref.m8"
    # mmseqs easy-search $query_path $target_path $save_path tmp --format-output "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch" --max-seqs 5000

    if not os.path.exists(alignment_path):
        process = subprocess.Popen(['mmseqs', 'easy-search', query_path, ref_seq_path, alignment_path, "tmp", "--format-output", "query,target,qaln,taln,qstart,qend,tstart,tend,mismatch"],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        # print(stdout)
        # print(stderr)

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_fasta_path', default="", type=str)
    parser.add_argument('--test_fasta_path', default="", type=str)
    parser.add_argument('--ckpt_saving_dir', default="", type=str)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    
    parser.add_argument('--min_testing_time', default=None, type=int)
    parser.add_argument('--max_testing_time', default=None, type=int)

    parser.add_argument('--epoch_num', default=500, type=int)

    parser.add_argument('--test_result_saving_path', default="", type=str)
    args = parser.parse_args()

    if "h3n2" in args.train_fasta_path:
        args.ref_seq_path = "../data/reference_fasta/prortein_A_NewYork_392_2004_H3N2_ha.fasta"
    elif "h1n1" in args.train_fasta_path:
        args.ref_seq_path = "../data/reference_fasta/protein_A_California_07_2009_H1N1_ha.fasta"
    
    args.train_alignment_path = args.train_fasta_path.split(".fasta")[0] + ".ref.m8"
    args.test_alignment_path = args.test_fasta_path.split(".fasta")[0] + ".ref.m8"

    return args

def build_tensors(dataset, aa_dict, seq_len):
    input_tensor = torch.zeros(len(dataset), seq_len)
    time_tensor = torch.zeros(len(dataset), 1)
    weight_tensor = torch.zeros(len(dataset), 1)
    for i, data in enumerate(dataset):
        for j, aa in enumerate(data["seq"]):
            input_tensor[i, j] = aa_dict.get(aa, aa_dict["<unk>"])

        time_tensor[i] = data["time_bin"]
        weight_tensor[i] = data["freq"] * data["bin_size"]
    return input_tensor.long(), time_tensor, weight_tensor
    
class Model(nn.Module):
    def __init__(self, seq_len, vocab) -> None:
        super().__init__()
        alpha = torch.randn(seq_len, len(vocab)) # [L, V]
        beta = torch.randn(seq_len, len(vocab)) # [L, V]

        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)

    def forward(self, input, times, loss_weight, mode="train"):
        # input: [B, L]
        # times: [B, 1]

        logits = self.alpha.unsqueeze(0).repeat(input.size(0), 1, 1) * times.unsqueeze(-1) + self.beta.unsqueeze(0).repeat(input.size(0), 1, 1)
        log_probs = torch.log_softmax(logits, dim=-1)
        # print(log_probs.size()) # [B, L, V]
        log_prob = torch.gather(log_probs, -1, input.unsqueeze(-1)).squeeze(-1)
        # print(log_prob.size())
        
        
        # print(nll)
        if mode == "train":
            nll = - log_prob.mean(-1) # [B]
            loss = torch.sum(nll * loss_weight.squeeze(-1)) / torch.sum(loss_weight.squeeze(-1))
            return loss
        nll = - log_prob.sum(-1) # [B]
        return nll

    def infer(self, times):
        prob = self.alpha.unsqueeze(0) * times.view(-1, 1, 1) + self.beta.unsqueeze(0) # [T, L, V]
        log_prob = torch.log_softmax(prob, dim=-1) # [T, L, V]
        return log_prob

def build_temporal_dataset(dataset):
    temporal_seq2freq = dict()
    temporal_aa2freq = dict()

    for item in dataset:
        time_bin = item["time_bin"]
        seq = item["seq"]
        freq = item["freq"]
        if time_bin not in temporal_seq2freq:
            temporal_seq2freq[time_bin] = defaultdict(float)
        temporal_seq2freq[time_bin][seq] += freq
        
        if time_bin not in temporal_aa2freq:
            temporal_aa2freq[time_bin] = dict()
        for pos, aa in enumerate(seq):
            if pos not in temporal_aa2freq[time_bin]:
                temporal_aa2freq[time_bin][pos] = defaultdict(float)
            temporal_aa2freq[time_bin][pos][aa] += freq
    for time in temporal_seq2freq:
        assert abs(1 - sum(temporal_seq2freq[time].values())) < 1e-5
    for time in temporal_aa2freq:
        for pos in temporal_aa2freq[time]:
            assert abs(1 - sum(temporal_aa2freq[time][pos].values())) < 1e-5
    return temporal_seq2freq, temporal_aa2freq



def test(dataloader, model, reduce=True):
    model.eval()


    weights = []
    nlls = []
    with torch.no_grad():
        for input, time, weight in dataloader:
            input, time, weight = input.cuda(), time.cuda(), weight.cuda()
            nll = model(input, time, weight, mode="test")
            # print(nll.size())
            nlls.append(nll.cpu())
            weights.append(weight.cpu())

    nlls = torch.cat(nlls)
    weights = torch.cat(weights).squeeze(-1)
    # print(nlls.size(), weights.size())
    
    ave_nll = torch.sum(weights * nlls) / torch.sum(weights)
    if reduce:
        return {"ave_nll": ave_nll.item()}

    return nlls, {"ave_nll": ave_nll.item()}


def train(train_dataloader, valid_dataloader, model, optimizer, epoch_num=100, ckpt_dir=None, verbose=True):
    step = 0

    performances = []
    
    for epoch in tqdm(range(epoch_num)):
        model.train()
        # print("Epoch", epoch)
        for batch, (input, time, weight) in enumerate(train_dataloader):
            step += 1
            input, time, weight = input.cuda(), time.cuda(), weight.cuda()
            loss = model(input, time, weight, mode="train")

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # manage the best
        valid_loss = test(valid_dataloader, model)["ave_nll"]
        # if step % 100 == 0:
        if verbose:
            print("Epoch %d, train loss: %g, valid_loss (nll): %g" % (epoch, loss.item(), valid_loss))
        
        if len(performances) == 0 or valid_loss < min(performances):
            if verbose:
                print("Saving checkpoint to %s" % (os.path.join(ckpt_dir, "best.ckpt")))
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.ckpt"))
        performances.append(valid_loss)


# def save_results(nlls, test_dataset, save_path):
    # ,src_id,prediction


if __name__ == "__main__":
    args = parse_args()

    print(args)

    ref_seq = str(next(SeqIO.parse(args.ref_seq_path, "fasta")).seq)
    proteinseq_toks = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', "<unk>"]
    vocab = {x: i for i, x in enumerate(proteinseq_toks)}

    batch_size=128
    model = Model(len(ref_seq), vocab).cuda()

    if not args.test:
        train_dataset = build_dataset(args.train_fasta_path, args.train_alignment_path, args.ref_seq_path)

        train_input_tensor, train_time_tensor, train_weight_tensor = build_tensors(train_dataset, vocab, len(ref_seq))
        # print(train_input_tensor.size(), train_time_tensor.size(), train_weight_tensor.size())

        full_dataset = TensorDataset(train_input_tensor, train_time_tensor, train_weight_tensor)
        valid_size = int(len(full_dataset) * 0.1)
        train_size = len(full_dataset) - valid_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, valid_size])
        # print(len(train_dataset), len(val_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(train_dataloader, valid_dataloader, model, optimizer, ckpt_dir=args.ckpt_saving_dir, verbose=args.verbose, epoch_num=args.epoch_num)

    if args.test_fasta_path and args.test_alignment_path:
        # Testing for the best model
        test_dataset_ori = build_dataset(args.test_fasta_path, args.test_alignment_path, args.ref_seq_path)

        test_input_tensor, test_time_tensor, test_weight_tensor = build_tensors(test_dataset_ori, vocab, len(ref_seq))
        # print(test_input_tensor.size(), test_time_tensor.size(), test_weight_tensor.size())
        if args.min_testing_time is not None and args.max_testing_time is not None:
            test_time_tensor = torch.arange(args.min_testing_time, args.max_testing_time + 1).unsqueeze(1).repeat(1, test_input_tensor.size(0)).view(-1, 1)
            test_weight_tensor = test_weight_tensor.repeat(args.max_testing_time - args.min_testing_time + 1, 1)
            test_input_tensor = test_input_tensor.repeat(args.max_testing_time - args.min_testing_time + 1, 1)
            # print(test_input_tensor.size(), test_time_tensor.size(), test_weight_tensor.size())
            # print(test_weight_tensor)
            # print(test_time_tensor)
        
        
        test_dataset = TensorDataset(test_input_tensor, test_time_tensor, test_weight_tensor)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model.load_state_dict(torch.load(os.path.join(args.ckpt_saving_dir, "best.ckpt")))
        nlls, test_loss = test(test_dataloader, model, reduce=False)
        print("================= Testing Results =================")
        for key in test_loss:
            print(key, test_loss[key])
    
        # src_id,prediction
        print("Saving results to %s" % args.test_result_saving_path)
        d = {'src_id': [x["id"] for x in test_dataset_ori] * (args.max_testing_time - args.min_testing_time + 1), 'prediction': [x.item() for x in nlls]}
        df = pd.DataFrame(data=d)
        if not os.path.exists(os.path.split(args.test_result_saving_path)[0]):
            os.makedirs(os.path.split(args.test_result_saving_path)[0])
        df.to_csv(args.test_result_saving_path)