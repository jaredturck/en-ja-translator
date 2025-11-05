import torch, os, csv, time, datetime, sys, re, platform, math
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Module
from torch.utils.data import Dataset

TRAIN_PATH = 'datasets/JSP/'
WEIGHTS_PATH = 'weights/'
JA_LENGTH = 21_592
VOCAB_SIZE = 259
MAX_EMB = 2048
TARGET_LOSS = 0.01
DEVICE = 'cuda'
TESTING_CAP = False

BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

if platform.uname().node == 'Jared-PC':
    BATCH_SIZE = 10
else:
    BATCH_SIZE = 35

class JATokenizer:
    ''' Byte tokenization for English and Japanese '''
    def __init__(self):
        self.ja_tokens = {'<BOS>' : BOS_ID, '<EOS>' : EOS_ID, '<UNK>' : UNK_ID}
        self.en_tokens = {'<BOS>' : BOS_ID, '<EOS>' : EOS_ID, '<UNK>' : UNK_ID}
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
        self.ja_ignore = re.compile(r'[\ue2fb\ue285\ue2fa\ue035\ue4c6\ue021\ue025\ue2f0\ue2f9\ue029\ue0fd\x95\x7f\ue045\x8d]')

        ja_count = 4
        for charset in self.symbol_ranges:
            for num in charset:
                self.ja_tokens[chr(num)] = ja_count
                ja_count += 1
        
        en_count = 4
        for num in range(0x20, 0x7E + 1): # English ASCII
            self.en_tokens[chr(num)] = en_count
            en_count += 1

        self.rja_tokens = {v : k for k,v in self.ja_tokens.items()}

    def ja_tokenize(self, txt):
        ''' Convert Japanese text to IDs '''
        txt = self.ja_ignore.sub('', txt)
        return [BOS_ID] + [self.ja_tokens.get(c, UNK_ID) for c in txt] + [EOS_ID]

    def en_tokenize(self, txt):
        ''' Convert English text to IDs '''
        return [BOS_ID] + [self.en_tokens.get(c, UNK_ID) for c in txt] + [EOS_ID]
    
    def ja_detokenize(self, ids):
        ''' Convert IDs to Japanese text '''
        return [self.rja_tokens.get(i, '<UNK>') for i in ids]

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
        no_tokens = 0
        for file in os.listdir(TRAIN_PATH):
            with open(os.path.join(TRAIN_PATH, file), 'r', encoding='UTF-8') as file:
                reader = csv.reader(file, delimiter=',')
                for row in reader:
                    en, ja = row[1], row[2]
                    en_ids = self.tokenizer.en_tokenize(en)[:MAX_EMB]
                    ja_ids = self.tokenizer.ja_tokenize(ja)[:MAX_EMB]
                    self.samples.append((torch.tensor(en_ids), torch.tensor(ja_ids)))
                    no_tokens += len(en_ids) + len(ja_ids)

                    if time.time() - start > 10:
                        start = time.time()
                        print(f'[+] Processed {len(self.samples):,} samples')
                        if TESTING_CAP and len(self.samples) >= 500_000:
                            break

        print(f'[+] Read {len(self.samples):,} samples ({no_tokens:,} tokens)')
    
    def collate_fn(self, batch):
        x,y = zip(*batch)
        x = pad_sequence(x, batch_first=True, padding_value=0)
        y = pad_sequence(y, batch_first=True, padding_value=0)
        return x,y

class EN2JAModel(Module):
    def __init__(self):
        super().__init__()
        self.d_model = 384
        self.nhead = self.d_model // 64
        self.dim_feedforward = self.d_model * 4
        self.num_layers = self.d_model // 128
        self.dropout = 0.1
        self.dataset = EN2JADataset()

        self.en_embedding = nn.Embedding(VOCAB_SIZE + 1, self.d_model, padding_idx=0)
        self.en_dropout = nn.Dropout(self.dropout)

        self.pos_embedding_en = nn.Embedding(MAX_EMB + 1, self.d_model, padding_idx=0)
        self.pos_embedding_ja = nn.Embedding(MAX_EMB + 1, self.d_model, padding_idx=0)

        self.encoder = nn.TransformerEncoder(
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

        self.ja_embedding = nn.Embedding(JA_LENGTH + 1, self.d_model, padding_idx=0)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=self.num_layers
        )

        self.adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features=self.d_model,
            n_classes=JA_LENGTH+1,
            cutoffs=[2000, 10_000],
            div_value=4.0
        )
    
    def forward(self, x, y):

        Bx, Sx = x.shape
        By, Sy = y.shape

        pos_x = self.pos_embedding_en(torch.arange(Sx, device=DEVICE)).unsqueeze(0).expand(Bx, Sx, -1)
        pos_y = self.pos_embedding_ja(torch.arange(Sy, device=DEVICE)).unsqueeze(0).expand(By, Sy, -1)

        memory = self.encoder(
            self.en_dropout(
                self.en_embedding(x) + pos_x
            ),
            src_key_padding_mask = (x == 0)
        )

        causal_mask = torch.triu(torch.ones(Sy, Sy, dtype=torch.bool, device=DEVICE), diagonal=1)

        decoder_out = self.decoder(
            self.en_dropout(
                self.ja_embedding(y) + pos_y,
            ),
            memory,
            tgt_mask = causal_mask,
            tgt_key_padding_mask = (y == 0),
            memory_key_padding_mask = (x == 0)
        )

        return decoder_out
    
    def train_model(self):
        self.load_weights()
        self.dataset.read_data()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.dataset.collate_fn)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        start = time.time()
        save_time = time.time()

        print('[+] Training Started')
        for epoch in range(1000):
            total_loss = 0
            for n,(src,tgt) in enumerate(self.dataloader):

                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)

                y_in = tgt[:, :-1]
                y_out = tgt[:, 1:]

                self.optimizer.zero_grad()

                out = self.forward(src, y_in)
                out2d = out.reshape(-1, out.size(-1))
                tgt1d = y_out.reshape(-1)
                mask = tgt1d != 0
                out = self.adaptive_softmax(out2d[mask], tgt1d[mask])
                loss = out.loss

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if time.time() - start > 10:
                    start = time.time()
                    print(f'Epoch {epoch+1}, batch {n+1:,} of {len(self.dataloader):,}, loss {loss.item():.4f}')

                    if time.time() - save_time > 600:
                        self.save_weights()
                        save_time = time.time()

            avg_loss = total_loss / len(self.dataloader)
            if avg_loss <= TARGET_LOSS:
                print(f'[+] Target loss reached')
                
            print(f'Epoch {epoch+1}, avg loss {avg_loss:.4f}')
    
    def save_weights(self):
        fname = os.path.join(WEIGHTS_PATH, f'weights_{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")}.pt')
        torch.save(self.state_dict(), fname)
        print(f'[+] Weights saved {fname}')
    
    def load_weights(self):
        files = [os.path.join(WEIGHTS_PATH, file) for file in os.listdir(WEIGHTS_PATH) if file.endswith('.pt')]
        if files:
            latest_file = max(files, key=os.path.getctime)
            self.load_state_dict(torch.load(latest_file))
            print(f'[+] Loaded weights {latest_file}')
    
    def predict(self, txt):
        self.eval()

        en_ids = self.dataset.tokenizer.en_tokenize(txt)[:MAX_EMB]
        en = torch.tensor(en_ids, device=DEVICE).unsqueeze(0)

        Bx, Sx = en.shape
        pos_x = self.pos_embedding_en(torch.arange(Sx, device=DEVICE)).unsqueeze(0).expand(Bx, Sx, -1)
        memory = self.encoder(
            self.en_dropout(self.en_embedding(en) + pos_x),
            src_key_padding_mask = (en == 0)
        )

        ja_seq = torch.tensor([[BOS_ID]], device=DEVICE)
        conf = []

        for _ in range(MAX_EMB):
            By, Sy = ja_seq.shape
            pos_y = self.pos_embedding_ja(torch.arange(Sy, device=DEVICE)).unsqueeze(0).expand(By, Sy, -1)
            causal_mask = torch.triu(torch.ones(Sy, Sy, dtype=torch.bool, device=DEVICE), diagonal=1)

            dec_out = self.decoder(
                self.en_dropout(self.ja_embedding(ja_seq) + pos_y),
                memory,
                tgt_mask = causal_mask,
                tgt_key_padding_mask = (ja_seq == 0),
                memory_key_padding_mask = (en == 0)
            )

            last_h = dec_out[:, -1, :]
            log_probs = self.adaptive_softmax.log_prob(last_h)
            next_token = torch.argmax(log_probs, dim=-1, keepdim=True)
            ja_seq = torch.cat([ja_seq, next_token], dim=1)
            conf.append(log_probs[0, next_token.item()].item())

            if next_token.item() == EOS_ID:
                break
        
        output = ''.join(self.dataset.tokenizer.ja_detokenize(ja_seq.tolist()[0][1:]))
        token_conf = [round(math.exp(i),2) for i in conf]
        overal_conf = round(math.exp(sum(conf) / len(conf)),2)
        print([output])
        print([token_conf, overal_conf])

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        try:
            model = EN2JAModel().to(DEVICE)
            model.train_model()
        except KeyboardInterrupt:
            model.save_weights()
    else:
        model = EN2JAModel().to(DEVICE)
        model.load_weights()
        while True:
            txt = input('> ')
            model.predict(txt)
