import torch
import matplotlib.pyplot as plt
import numpy
import torch.nn.functional as F
import torch.nn as nn

# downloading our dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Hyperparmeters
batch_size = 64
block_size = 256
max_iters = 3000
eval_intervals = 300
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda'if torch.cuda.is_available() else 'cpu'


# some functions to load our dataset, create test-val sets and split into Batches
def load_dataset(path='input.txt'):
    with open(path, 'r',encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[n] for n in l])

    data = torch.tensor(encode(text), dtype=torch.long)

    # Split once and return both splits
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return {
        'train_data': train_data,
        'val_data': val_data,
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
        'encode': encode,
        'decode': decode
    }


def get_batch(data, split, block_size, batch_size, device):
    split_data = data['train_data'] if split == 'train' else data['val_data']
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i:i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)



    # Lets define the different blocks of our transformer
class Head(nn.Module):
    '''One head of self-attention'''

    def __init__(self,n_embd, head_size,block_size,dropout) -> None:
        super().__init__()
        self.head_size = head_size
        # initialse each q,k,v vector as (n_embd,head_size) as we want each of these to interact with each token which itself is represented by a (65dimension vector) 1 T = 65, so anything that
        # needs to interact with 1 T must be, 65 by x
        self.key = nn.Linear(n_embd,head_size,bias=False) 
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self,x):
        B,T,C = x.shape # B,T,C
        k = self.key(x)  # B,T,C 
        q = self.query(x) # B,T,C
        v = self.value(x) # B,T,C

        wei = q @ k.transpose(-2,-1) * (self.head_size**-0.5) # B,T,C @ B,C,T => # B,T,T #we also a add a normalisation term 
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # type: ignore # B,T,T
        wei = F.softmax(wei,dim=1) # B,T,T
        wei = self.dropout(wei)

        out = wei @ v # B,T,T @ B,T,C => B,T,C

        return out        

class MultiHeadAttention(nn.Module):
    '''Multiple heads of self-attention'''

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out)) #we project because we want to fork-off to do some computation, then merge it into the residual connection
        return out
class FeedForward(nn.Module):
    '''Simple linear layer followed by a non-linearity'''
    # FF applies a mini brain only to that token, with no interaction with others
    # Think of attention as "talking to your friends", and FeedForward as "processing that info alone in your head."
    # expands, activates, compresses. Like thinking hard then summarizing.

    def __init__(self, n_embd,dropout) -> None:
        super().__init__()
        # expands, activates, compresses. Like thinking hard then summarizing.
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd), # After the tokens have communicated via attention we want the tokens to think, and to do it we project the embedding into a 4X dimension vector , basically increase the computation width
            nn.ReLU(),  # Non-linearity to capture complex interactions
            nn.Linear(4*n_embd,n_embd), # And then collapse it back.
            nn.Dropout(dropout)
        )

    def forward(self,x):
        out = self.net(x)
        return out

class Block(nn.Module):
    '''Simple linear layer followed by a non-linearity'''
    
    def __init__(self,n_embd,n_head,block_size,dropout) -> None:
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_embd,n_head,block_size,dropout) #i.e 4 heads of 8-dimensional heads
        self.ffwd = FeedForward(n_embd,dropout)
        self.ln1 = nn.LayerNorm(n_embd) #prenorm-formulation
        self.ln2 = nn.LayerNorm(n_embd) #layernorm is done per-feature, i.e mean and variance is calculated each token across 32dim vectors
    
    def forward(self,x):
        x = x + self.sa(self.ln1(x)) #intiating skip connections
        x = x + self.ffwd(self.ln2(x)) #intiating skip connections 
        return x

class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size,n_embd,n_head,block_size,dropout) -> None:
        super().__init__()
        # we embedd the tokens
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) #n_embd = C 
        self.position_embedding_table =  nn.Embedding(block_size ,n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd,n_head,block_size,dropout),
            Block(n_embd,n_head,block_size,dropout),
            Block(n_embd,n_head,block_size,dropout),
            nn.LayerNorm(n_embd)
        )
        self.lm_head = nn.Linear(n_embd,vocab_size) #the last linear layer to map back to vocab size

    def forward(self,idx,targets=None):
        B , T = idx.shape
        # we grab the corresponding embeddings
        token_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T,device=device))
        x = token_embd + pos_embd #addition as we're just adding in the context of position
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)

            loss = nn.functional.cross_entropy(logits,targets)
        
        return logits,loss
    
    def generate(self,idx,max_tokens):
        for _ in range(max_tokens):
            # when our generation grows longer than block_size, we would still want to feed in 8 characters at a time, as this ensures we never run out of scope in our embedding table which is (block_size ,n_embd)
            # so we clip the idx, to only contain the last 8 tokens or whatever blocks_size we chose
            idx_cond = idx[:, -block_size:] 
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:] # only get the last logit
            probs = nn.functional.softmax(logits,dim=-1)
            idnext = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idnext),dim=1)
        return idx

    # def __repr__(self):
    #     return f'BigramLanguageModel with {self.token_embedding_table}'

def main():
    # loading and encoding the dataset
    dataset = load_dataset('input.txt')
    vocab_size = dataset['vocab_size']
    encode = dataset['encode']
    decode = dataset['decode']
    data = dataset
    xb, yb = get_batch(data, split='train', block_size=block_size, batch_size=batch_size, device=device)


    # instantiating the model
    m = BigramLanguageModel(vocab_size,n_embd,n_head,block_size,dropout)
    model = m.to(device)
    out,loss = model(xb,yb)
    print(out.shape)
    print(f'{loss.item():.4f}')

    # Backward pass and optimise
    optimizer = torch.optim.AdamW(m.parameters(),lr = learning_rate)

    for steps in range(10000):
        logits,loss = m(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(loss.item())


    # inference
    ix = torch.zeros((1,1),dtype=torch.long,device=device)
    print((decode(m.generate(ix,500)[0].tolist())))

if __name__=='__main__':
    main()