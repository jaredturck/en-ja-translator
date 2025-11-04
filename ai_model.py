import torch, os, csv, time, datetime, sys
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import Dataset
from transformers import AutoTokenizer

TRAIN_PATH = 'datasets/JSP/'
WEIGHTS_PATH = 'weights/'
EN_LENGTH = 512
JA_LENGTH = 128
DEVICE = 'cuda'
BATCH_SIZE = 256

class EN2JADataset(Dataset):
    def __init__(self):
        self.samples = []
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

    def __len__(self):
        return len(self.samples)
    
    def __get__item__(self, idx):
        return self.samples[idx]
    
    def read_data(self):
        start = time.time()
        for file in os.listdir(TRAIN_PATH):
            with open(os.path.join(TRAIN_PATH, file), 'r', encoding='UTF-8') as file:
                reader = csv.reader(file, delimiter=',')
                for row in reader:
                    en, ja = row[0], row[1]
                    en_ids = self.tokenizer(en, return_tensors='pt').input_ids
                    ja_ids = self.tokenizer(ja, return_tensors='pt').input_ids
                    self.samples.append((en_ids, ja_ids))

                    if time.time() - start > 10:
                        start = time.time()
                        print(f'[+] Processed {len(self.samples):,} samples')
    
    def collate_fn(self, batch):
        return batch

class EN2JAModel(Module):
    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.nhead = self.d_model // 64
        self.dim_feedforward = self.d_model * 4
        self.num_layers = self.d_model // 128
        self.dropout = 0.05
        self.dataset = EN2JADataset()

        self.en_embedding = nn.Embedding(EN_LENGTH, self.d_model)
        self.en_dropout = nn.Dropout(self.dropout)
        self.pos_embedding = nn.Embedding(EN_LENGTH, self.d_model)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=self.num_layers
        )
        self.ja_proj = nn.Linear(self.d_model, JA_LENGTH)
    
    def forward(self, x):

        B, S = x.shape
        pos = self.pos_embedding(torch.arange(S, device=DEVICE)).unsqueeze(0).expand(B, S, -1)
        
        out = self.ja_proj(
            self.transformer(
                self.en_dropout(
                    self.en_embedding(x) + pos
                )
            )
        )
        return out
    
    def train_model(self):
        self.dataset.read_data()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.dataset.collate_fn)
        self.optmizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        loss_func = nn.CrossEntropyLoss()
        start = time.time()
        save_time = time.time()

        print('[+] Training Started')
        for epoch in range(1000):
            total_loss = 0
            for n,(src,tgt) in enumerate(self.dataloader):
                B,T,S = src.shape
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)

                out = self.forward(src)
                loss = loss_func(out, tgt)

                loss.backward()
                self.optmizer.step()
                total_loss += loss.item()

                if time.time() - start > 10:
                    start = time.time()
                    print(f'Epoch {epoch+1}, batch {n+1} of {len(self.dataloader)}, loss {loss.item():.4f}')

                    if time.time() - save_time > 600:
                        self.save_weights()
                        save_time = time.time()
                
            print(f'Epoch {epoch+1}, avg loss {total_loss / len(self.dataloader):.4f}')
    
    def save_weights(self):
        torch.save(self.state_dict(), os.path.join(WEIGHTS_PATH, f'weights_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%S")}.pt'))

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        try:
            model = EN2JAModel().to(DEVICE)
            model.train_model()
        except KeyboardInterrupt:
            model.save_weights()
