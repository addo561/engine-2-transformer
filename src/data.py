import torch
from torch import nn
import tiktoken
from torch.utils.data import DataLoader,Dataset
import os
from bs4 import BeautifulSoup
import requests

###GET TEXT DATA(GET_DATA FUNCTION),BUILD DATALOADER FUNCS AND DATASET  CLASS

#SET URL
def get_data(text_file, save : bool = False):
    '''returns txt  file  of  raw  text'''
    path = os.path.join('.',text_file)

    url  =  'https://en.wikisource.org/wiki/A_Journey'
    headers =  {
        'User-Agent':'DataExtractionScript/1.0 (korlilarryaddo43@gmail.com)'
    }
    res = requests.get(url,headers=headers)
    if res.status_code==200: #check if request was  succesfull
        soup =  BeautifulSoup(res.text,'html.parser')
        # Strip extra whitespace from each paragraph, remove sometexts
        texts = [t.get_text().strip().replace('\n',' ') for t in  soup.find_all('p')  if t.get_text().strip() ][1:-3]
        #seperate with new line
        texts  = ' '.join(texts)
        words = texts.split()
        new = ''
        for w in words:
            new += w + ' '
            if '.' in w :
                new += '\n'
        #dump to TEXT_FILE
        if save:
            with  open(path,'w')  as f:
                f.writelines(new)   
    else:
        print('bad request')
    return new

get_data('text.txt',True)

### TOKENIZATION FROM MODEL DATASET  AND DATALOADER

class Tdataset(Dataset):
    def __init__(self,txt,tokenizer,max_length,stride):
        self.X = []
        self.y = []

        #get  ids of textswith tokenizer
        token_ids = tokenizer.encode(txt,allowed_special={"<|endoftext|>"})
        #get inputs and outputs wiht step of 1
        for i in range(0,len(token_ids) - max_length,stride):
            inputs = token_ids[i:i+max_length]
            outs = token_ids[i+1:i+max_length+1]
            self.X.append(torch.tensor(inputs))
            self.y.append(torch.tensor(outs))
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.X[index],self.y[index]
    
def loader(txt,max_length,stride,batch_size,
           shuffle=True,num_workers=0,drop_last = True):
    #tokenizer defined
    tokenizer =  tiktoken.get_encoding('gpt2') 
    #get dataset
    dataset = Tdataset(txt,tokenizer,max_length,stride)
    
    #create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return dataloader


### FUNC FOR INPUT EMBEDDING, COMBINE POSITIONAL AND TOKEN_EMBEDDING

def get_input_embedding(txt,
                        max_length,
                        stride,
                        batch_size,
                        num_workers
                        ):
    #POSITIONAL AND TOKEN EMBEDDING
    vocab_size = 50257
    out_dim = 256

    token_embed_layer = nn.Embedding(vocab_size,out_dim)#set shapes 
    positional_embed_layer = nn.Embedding(max_length,out_dim)#

    dataloader =loader(txt,
                       max_length=max_length,
                       stride=stride,
                       batch_size=batch_size,
                       num_workers=num_workers,  
                       )
    for X,y in dataloader:
        token_embeddings = token_embed_layer(X)
        positional_embeddings = positional_embed_layer(torch.arange(max_length))
        input_embeddings = token_embeddings + positional_embeddings
        break
    return input_embeddings