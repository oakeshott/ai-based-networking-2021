import json
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import joblib
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data.dataloader import default_collate
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="Preprocessing")
parser.add_argument('-i', '--input', default='similarity/train',
        help='Input directory')
parser.add_argument('-l', '--log', default='logs',
        help='Log directory')
parser.add_argument('-o', '--model_dir', default='models',
        help='Model directory')
parser.add_argument('--model-path', default='model.mdl',
        help='Model path')
parser.add_argument('-b', '--batchsize', default=32, type=int,
        help='Log directory')
parser.add_argument('--log-seq', default=1, type=int,
        help='Log sequence')
parser.add_argument('--sequence-length', default=1000, type=int,
        help='Sequence length')
parser.add_argument('--fold', default=4, type=int,
        help='Fold split')
parser.add_argument('--seed', default=1, type=int,
        help='seed')
parser.add_argument('--test', action="store_true",
        help='test mode')
parser.add_argument('--train', action="store_true",
        help='train mode')

args = parser.parse_args()

class NetworkStateEstimationFromVideoStreaming(nn.Module):
    def __init__(self, input_dim, target_dim, sequence_length, hidden_dim):
        super(NetworkStateEstimationFromVideoStreaming, self).__init__()
        self.input_dim       = input_dim
        self.target_dim      = target_dim
        self.sequence_length = sequence_length
        self.hidden_dim      = hidden_dim

        self.fc1     = nn.Linear(self.input_dim * self.sequence_length, self.hidden_dim)
        self.fc2     = nn.Linear(self.hidden_dim, self.target_dim)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x   = x.view(-1, self.input_dim * self.sequence_length)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class Transformer:
    def __init__(self, is_train=False):
        self.transformer = dict()
        self.is_train = is_train
        if not self.is_train:
            self.transformer = self.load()

    def __call__(self, df, metric):
        if self.is_train:
            return self.fit_transform(df, metric)
        else:
            return self.transformer[metric].transform(df)

    def inverse_transform(self, scaled, metric):
        return self.transformer[metric].inverse_transform(scaled)

    def fit_transform(self, df, metric):
        self.transformer[metric] = MinMaxScaler()
        df = self.transformer[metric].fit_transform(df)
        self.dump()
        return df

    def dump(self, filename='/tmp/transformer.bin'):
        with open(filename, 'wb') as f:
            joblib.dump(self.transformer, f)

    def load(self, filename='/tmp/transformer.bin'):
        with open(filename, 'rb') as f:
            data = joblib.load(f)
        return data

class SimilarityDataset(Dataset):
    def __init__(self, path, sequence_length, device=None, transform=None, masks=None, is_train=True):
        self.sequence_length = sequence_length
        self.transform = transform

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        metrics        = ["psnr", "ssim"]
        target_metrics = ["throughput", "loss_rate"]

        self.input_dim  = len(metrics)
        self.target_dim = len(target_metrics)

        if is_train:
            files = glob.glob(os.path.join(path, "*.json"))
            df = self.read_files(files, self.sequence_length)
        else:
            df = self.read_file(path, sequence_length)

        # if self.transform:
        #     for metric in metrics:
        #         df[[metric]] = self.transform(df[[metric]], metric)

        df.index   = df[["video_type", "throughput", "loss_rate", "interval"]]
        indices    = df[["video_type", "throughput", "loss_rate", "interval"]].index.unique()
        grouped_df = df.groupby(["video_type", "throughput", "loss_rate", "interval"])

        self.data   = []
        self.target = []
        for index in tqdm(indices):
            series = grouped_df.get_group(index)
            if len(series) < self.sequence_length:
                continue
            video_type, throughput, loss_rate, interval = index
            self.data.append(series[metrics].values)
            self.target.append([throughput, loss_rate])

    def __getitem__(self, idx):
        ret = torch.tensor(self.data[idx], dtype=torch.float64, device=self.device)
        trg = torch.tensor(self.target[idx], dtype=torch.float64, device=self.device)
        return ret, trg

    def __len__(self):
        return len(self.data)

    def read_files(self, files, sequence_length):
        li_df = []
        for filename in tqdm(files):
            df = self.read_file(filename, sequence_length)
            li_df.append(df)
        df = pd.concat(li_df)
        return df

    def read_file(self, filename, sequence_length):
        with open(filename) as f:
            d = json.load(f)
        df = pd.DataFrame.from_dict(d)
        df["interval"] = df["frame_index"] // sequence_length
        df = df.sort_values("frame_index")
        return df


def collate_fn(batch):
    data, targets = list(zip(*batch))
    data    = torch.stack(data)
    targets = torch.stack(targets)
    return data, targets

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformer = None
    # transformer = Transformer(is_train=True)

    similarity_dir = args.input
    log_dir        = args.log
    model_dir      = args.model_dir
    seed           = args.seed
    fold_split     = args.fold
    batch_size     = args.batchsize
    log_seq        = args.log_seq

    set_seed(seed)

    sequence_length = args.sequence_length
    hidden_dim      = 128
    num_layers      = 2
    max_epoches     = 2000

    train_and_valid_dataset = SimilarityDataset(
            path=similarity_dir,
            sequence_length=sequence_length,
            device=device,
            transform=transformer
            )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    writer = SummaryWriter(logdir=log_dir)

    X = train_and_valid_dataset.data
    Y = train_and_valid_dataset.target
    skf = KFold(n_splits=fold_split, shuffle=True, random_state=seed)
    for i, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):
        train_dataset = Subset(train_and_valid_dataset, train_idx)
        valid_dataset = Subset(train_and_valid_dataset, valid_idx)
        train_size = len(train_dataset)
        valid_size = len(valid_dataset)
        print(f'fold : {i+1} train dataset size : {train_size} valid dataset size: {valid_size}')

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_dataloader = DataLoader(valid_dataset, batch_size=valid_size, shuffle=True, collate_fn=collate_fn)

        input_dim  = train_and_valid_dataset.input_dim
        target_dim = train_and_valid_dataset.target_dim

        model = NetworkStateEstimationFromVideoStreaming(input_dim, target_dim, sequence_length, hidden_dim).to(device)

        loss_function = nn.L1Loss()
        l1_loss_function = nn.L1Loss()

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(1, max_epoches + 1):
            train_loss = 0
            for train_inputs, train_targets in train_dataloader:

                train_inputs  = train_inputs.float()
                train_targets = train_targets.float()
                train_inputs  = train_inputs.to(device)
                train_targets = train_targets.to(device)

                optimizer.zero_grad()

                train_scores = model(train_inputs)

                loss = loss_function(train_scores, train_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            train_loss /= len(train_dataloader)

            with torch.no_grad():
                valid_inputs, valid_targets = iter(valid_dataloader).next()

                valid_inputs  = valid_inputs.float()
                valid_targets = valid_targets.float()
                valid_inputs  = valid_inputs.to(device)
                valid_targets = valid_targets.to(device)

                valid_scores = model(valid_inputs)

                loss = loss_function(valid_scores, valid_targets)
                valid_loss = loss.item() / len(valid_dataloader)

                val_scores  = valid_scores.to('cpu').detach().numpy().astype(np.float32)
                val_targets = valid_targets.to('cpu').detach().numpy().astype(np.float32)

                throughput_scores = valid_scores[:, 0]
                loss_rate_scores  = valid_scores[:, 1]
                throughput_targets = valid_targets[:, 0]
                loss_rate_targets  = valid_targets[:, 1]

                throughput_loss, loss_rate_loss = (l1_loss_function(throughput_scores, throughput_targets).item(), l1_loss_function(loss_rate_scores, loss_rate_targets).item())

            writer.add_scalars(f"train_fold{i+1}", {
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "throughput_loss": throughput_loss,
                "packet_loss_rate_loss": loss_rate_loss
                }, epoch)

            if epoch <= 10 or epoch % log_seq == 0:
                print(f"Epoch: [{epoch}/{max_epoches}] train/valid loss: {train_loss:.4f} / {valid_loss:.4f} throughput/loss rate: {throughput_loss:.4f} / {loss_rate_loss:.4f}")

            if epoch % 100 == 0:
                torch.save(model.state_dict(), f"./{model_dir}/fold{i + 1}_{epoch}.mdl")

    writer.export_scalars_to_json(f"./{log_dir}/all_scalars.json")
    writer.close()

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer = None
    # transformer = Transformer(is_train=False)
    similarity_dir = args.input
    log_dir        = args.log
    model_dir      = args.model_dir
    model_path     = args.model_path
    seed           = args.seed
    batch_size     = args.batchsize

    set_seed(seed)

    sequence_length = args.sequence_length
    hidden_dim      = 128
    num_layers      = 2
    max_epoches     = 2000

    test_dataset = SimilarityDataset(
            path=similarity_dir,
            sequence_length=sequence_length,
            device=device,
            transform=transformer,
            is_train=False
            )

    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, collate_fn=collate_fn)

    input_dim  = test_dataset.input_dim
    target_dim = test_dataset.target_dim

    model = NetworkStateEstimationFromVideoStreaming(input_dim, target_dim, sequence_length, hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path))

    loss_function = nn.L1Loss()

    with torch.no_grad():
        test_inputs, test_targets = iter(test_dataloader).next()
        test_inputs  = test_inputs.float()
        test_targets = test_targets.float()

        test_scores = model(test_inputs)
        test_targets = test_targets.to(device)
        test_loss   = loss_function(test_scores, test_targets)

        throughput_scores = test_scores[:, 0]
        loss_rate_scores  = test_scores[:, 1]
        throughput_targets = test_targets[:, 0]
        loss_rate_targets  = test_targets[:, 1]

    throughput = throughput_scores.mean()
    loss_rate  = loss_rate_scores.mean()
    # throughput = transformer.inverse_transform(pd.DataFrame(throughput_scores), "throughput").squeeze().mean()
    # loss_rate  = transformer.inverse_transform(pd.DataFrame(loss_rate_scores), "loss_rate").squeeze().mean()

    print(f"input data: {similarity_dir} model: {model_path} throughput / loss rate: {throughput:.4f} / {loss_rate:.4f}")
    filename = "logs/result.txt"
    with open(filename, "a") as f:
        f.write(f"{similarity_dir}\t{model_path}\t{throughput:.4f}\t{loss_rate:.4f}\n")

def main():
    if args.train:
        train()
    if args.test:
        test()
if __name__ == '__main__':
    main()
