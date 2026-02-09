import torch
from torch import nn
# CLASSES ARE SELF_ATTENTION,CASUAL ONE WITH  MASKING AND MULTIHEADSELF ATTENION

class SelfAttention(nn.Module):
  # SIMPLE SELF ATTENTION WITH  TRAINABLE WEIGHTS 
  # NOO MASKING OF ATTENTION WEIGHTS
  def __init__(self,d_in,d_out,w_bias=False):
    super().__init__()
    self.W_query = nn.Linear(d_in,d_out,bias=w_bias)
    self.W_key = nn.Linear(d_in,d_out,bias=w_bias)
    self.W_value = nn.Linear(d_in,d_out,bias=w_bias)

  def forward(self,x):
    query = self.W_query(x)
    key = self.W_key(x)
    value = self.W_value(x)
    attn_scores = query @ key.T
    attn_weights = torch.softmax(attn_scores/key.shape[-1]**0.5,
                                 dim=-1)
    context_vec = attn_weights @ value
    return  context_vec

class Casual_self_Attention(nn.Module):
  #CASUAL SELFATTENTION WITH MASKING AND DROPOUT 
  def __init__(self,d_in,d_out,context_length,dropout,w_bias=False):
    super().__init__()
    self.W_query = nn.Linear(d_in,d_out,bias=w_bias)
    self.W_key = nn.Linear(d_in,d_out,bias=w_bias)
    self.W_value = nn.Linear(d_in,d_out,bias=w_bias)
    #apply dropout during  training
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

  def forward(self,x):
    #batch,num_tokens,input_embedding_dim (b,n,e)
    b,n,d_in = x.shape
    query = self.W_query(x) #shape (b,n,new_dim =  d_out),same for key and  value
    key = self.W_key(x)
    value = self.W_value(x)
    # b dim is maintained and multiplication happends accross n and n_dim 
    attn_scores = query @ key.transpose(1,2) # (b,n,n)
    # if number of tokens in batch is less than  context_size
    masked = attn_scores.masked_fill_(self.mask.bool()[:n][:n],-torch.inf)#(b,n,n)
    attn_weights = torch.softmax(masked/key.shape[-1]**0.5,
                                 dim=-1)#(b,n,n)
    drop_weights  = self.dropout(attn_weights)#(b,n,n)
    context_vec = drop_weights @ value #(b,n,d_out)
    return  context_vec    


class MultiheadSelfAttention(nn.Module):
    # MULTIHEAD SELFATTENTION
    # SPLIT Q,K,V INTO MATRICES  FOR EACH HEAD ,RATHER THAN STACKING THEM WITH  NN.MODULELIST
    # YOU  CAN  CONTROL D_OUT  
  def __init__(self,d_in,d_out,dropout,num_heads,context_length,qkv_bias=False):
    super().__init__()
    self.num_heads = num_heads
    # dimensions for each head, example if d_out  is 4  each dim head has d_out 2 if num_heads =  2
    self.head_dim =  d_out//num_heads 
    self.d_out = d_out

    self.W_query = nn.Linear(d_in,d_out,bias=qkv_bias)
    self.W_key = nn.Linear(d_in,d_out,bias=qkv_bias)
    self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
    #apply dropout during  training
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

  def forward(self,x):
    #(batch,tokens,input_embedding_dim)
    b,num_tokens,d_in = x.shape
    queries = self.W_query(x) #shape (b,n,new_dim =  d_out),same for key and  value
    keys = self.W_key(x)
    values = self.W_value(x)
    #split them into shape (b,tokens,num_heads,head_dim) 
    #Each  head has head_dim ,d_out//num_heads
    # Since d_out is num_heads *  head_dim
    queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)
    keys = keys.view(b,num_tokens,self.num_heads,self.head_dim)
    values = values.view(b,num_tokens,self.num_heads,self.head_dim)
    # Transpose  them  to shape (b,num_heads,tokens,head_dim)
    # Each head has a number of tokens with dim = head_dim
    # Cause you'll  want to work with  tokens with  corresponding dim for calculating  attn_scores
    queries = queries.transpose(1,2)
    keys = keys.transpose(1,2)
    values = values.transpose(1,2)
    # attn_scores for each head, scaled dot product attention
    # Shape (b,num_heads,T,T)
    attn_scores  = queries @ keys.transpose(2,3)

    # if number of tokens in batch is less than  context_size
    masked = attn_scores.masked_fill_(self.mask.bool()[:num_tokens][:num_tokens],-torch.inf)
    attn_weights = torch.softmax(masked/keys.shape[-1]**0.5,
                                 dim=-1)
    drop_weights  = self.dropout(attn_weights)
    # Shape (b,num_heads,Tokens,head_dim) -> (b,tokens,num_heads,head_dim) 
    context_vec = (drop_weights  @ values).transpose(1,2)
    #combine head ,d_out = num_heads * head_dim
    c_v  = context_vec.contiguous().view(b,num_tokens,self.d_out)
    return c_v


