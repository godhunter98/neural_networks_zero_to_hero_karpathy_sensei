import torch
import matplotlib.pyplot as plt
import numpy
import torch.nn.functional as F

# downloading our dataset
#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# creating and splitting the dataset
with open('./input.txt','r',encoding='utf-8') as f:
    text = f.read()
print("lenth of dataset is:",len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[n] for n in l])

data = torch.tensor(encode(text),dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


# Hyperparmeters
batch_size = 32
vocab_size = len(chars)
block_size = 8
max_iters = 3000
eval_intervals = 300
learning_rate = 1e-3
n_embd = 32

torch.manual_seed(42)
def get_batch(split:str):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+block_size+1]for i in ix])
    return x,y

xb,yb = get_batch('train')
print("X's",xb.shape)
print("Y's",yb.shape)


# Lets define the different blocks of our transformer
import torch.nn as nn

class Head(nn.Module):
    '''One head of self-attention'''

    def __init__(self, head_size) -> None:
        super().__init__()
        self.head_size = head_size
        # initialse each q,k,v vector as (n_embd,head_size) as we want each of these to interact with each token which itself is represented by a (65dimension vector) 1 T = 65, so anything that
        # needs to interact with 1 T must be, 65 by x
        self.key = nn.Linear(n_embd,head_size,bias=False) 
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C = x.shape # B,T,C
        k = self.key(x)  # B,T,C 
        q = self.query(x) # B,T,C
        v = self.value(x) # B,T,C

        wei = q @ k.transpose(-2,-1) * (C**-0.5) # B,T,C @ B,C,T => # B,T,T #we also a add a normalisation term 
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) # type: ignore # B,T,T
        wei = F.softmax(wei,dim=1) # B,T,T

        out = wei @ v # B,T,T @ B,T,C => B,T,C

        return out        

class MultiHeadAttention(nn.Module):
    '''Multiple heads of self-attention'''

    def __init__(self,num_heads,head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self,x):
        return torch.cat([h(x) for h in self.heads],dim=-1)
    
    
class FeedForward(nn.Module):
    '''Simple linear layer followed by a non-linearity'''
    # FF applies a mini brain only to that token, with no interaction with others
    # Think of attention as "talking to your friends", and FeedForward as "processing that info alone in your head."
    # expands, activates, compresses. Like thinking hard then summarizing.

    def __init__(self, n_embd) -> None:
        super().__init__()
        # expands, activates, compresses. Like thinking hard then summarizing.
        self.net = nn.Sequential(
            nn.Linear(n_embd,4*n_embd), # After the tokens have communicated via attention we want the tokens to think, and to do it we project the embedding into a 4X dimension vector
            nn.ReLU(),  # Non-linearity to capture complex interactions
            nn.Linear(4*n_embd,n_embd) # And then collapse it back.
        )

    def forward(self,x):
        out = self.net(x)
        return out

class Block(nn.Module):
    '''Simple linear layer followed by a non-linearity'''
    
    def __init__(self,n_embd,n_head) -> None:
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head,head_size) #i.e 4 heads of 8-dimensional heads
        self.ffwd = FeedForward(n_embd)
    
    def forward(self,x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        # we embedd the tokens
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) #n_embd = C 
        self.position_embedding_table =  nn.Embedding(block_size ,n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd,4),
            Block(n_embd,4),
            Block(n_embd,4)
        )
        self.lm_head = nn.Linear(n_embd,vocab_size) #the last linear layer to map back to vocab size

    def forward(self,idx,targets=None):
        B , T = idx.shape
        # we grab the corresponding embeddings
        token_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T))
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


# instantiating the model
m = BigramLanguageModel()
out,loss = m(xb,yb)
print(out.shape)
print(f'{loss.item():.4f}')

# Backward pass and optimise
optimizer = torch.optim.AdamW(m.parameters(),lr = 1e-3)

for steps in range(10000):

    xb,yb = get_batch('train')

    logits,loss = m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())


# inference
ix = torch.zeros((1,1),dtype=torch.long)
print((decode(m.generate(ix,500)[0].tolist())))