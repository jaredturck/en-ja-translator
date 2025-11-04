import torch, os, csv, time, datetime, sys, pickle, re
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import Dataset

TRAIN_PATH = 'datasets/JSP/train.csv'
WEIGHTS_PATH = 'weights/'
EN_LENGTH = 512
JA_LENGTH = 128
VOCAB_SIZE = 259
DEVICE = 'cuda'
BATCH_SIZE = 256

class JATokenizer:
    ''' Byte tokenization for English and Japanese '''
    def __init__(self):
        self.ja_tokens = {}
        self.en_tokens = {}
        self.symbol_ranges = [
            range(0x3040, 0x309F + 1), # Hiragana
            range(0x30A0, 0x30FF + 1), # Katakana
            range(0x4E00, 0x9FFF + 1), # Kanji
            range(0X3000, 0x303F + 1), # Punctuation
            range(0x20, 0x7E + 1),     # English ASCII
            range(0x2000, 0x206F + 1), # General punctuation
            [
                9, 10, 130, 161, 162, 163, 167, 174, 176, 177, 182, 183, 191, 201, 215, 224, 225, 227, 231, 232, 233, 234, 237, 241, 243, 247, 252, 283, 
                299, 769, 773, 945, 946, 947, 952, 956, 960, 963, 964, 969, 1567, 1570, 1574, 1575, 1576, 1578, 1581, 1582, 1583, 1585, 1586, 1587, 1588, 
                1593, 1601, 1602, 1604, 1605, 1606, 1607, 1608, 1705, 1711, 1740, 3618, 3619, 3629, 3656, 8361, 8364, 8592, 8593, 8594, 8658, 8704, 8711, 
                8722, 8734, 8744, 8810, 8811, 9472, 9473, 9484, 9488, 9609, 9632, 9633, 9650, 9651, 9660, 9661, 9670, 9671, 9675, 9678, 9679, 9733, 9734, 
                9742, 9786, 9792, 9794, 9825, 9829, 9834, 9835, 9836, 9837, 10084, 10145, 10629, 10630, 12579, 44036, 44192, 44208, 44277, 44284, 45716, 
                45796, 49884, 50864, 51060, 51221, 51316, 51452, 51648, 54616, 57410, 58349, 134071
            ] # extended symbols
        ]

        ja_count = 1
        for charset in self.symbol_ranges:
            for num in charset:
                self.ja_tokens[chr(num)] = ja_count
                ja_count += 1

        en_count = 1
        for num in range(0x20, 0x7E + 1): # English ASCII
            self.en_tokens[chr(num)] = en_count
            en_count += 1

    def ja_tokenize(self, txt):
        ''' Convert Japanese text to IDs '''
        txt = re.sub(r'[\ue2fb\ue285\ue2fa\ue035\ue4c6\ue021\ue025\ue2f0\ue2f9\ue029\ue0fd\x95\x7f\ue045\x8d]', '', txt)
        return [self.ja_tokens[c] for c in txt]

    def en_tokenize(self, txt):
        ''' Convert English text to IDs '''
        return [self.en_tokens[c] for c in txt]

class EN2JADataset(Dataset):
    def __init__(self):
        self.samples = []
        self.tokenizer = JATokenizer()

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def read_data(self):
        start = time.time()
        with open(TRAIN_PATH, 'r', encoding='UTF-8') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                en, ja = row[1], row[2]
                en_ids = self.tokenizer.en_tokenize(en)
                ja_ids = self.tokenizer.ja_tokenize(ja)
                self.samples.append((en_ids, ja_ids))

                if time.time() - start > 10:
                    start = time.time()
                    print(f'[+] Processed {len(self.samples):,} samples')

        with open(os.path.join('datasets', f'tensors_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%S")}.pt'), 'wb') as file:
            pickle.dump(self.samples, file)
    
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

        self.en_embedding = nn.Embedding(VOCAB_SIZE, self.d_model)
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
