import torch
from torch import nn
import tiktoken
from torch.utils.data import DataLoader,Dataset
import os
from bs4 import BeautifulSoup
import requests

###GET TEXT DATA(GET_DATA FUNCTION),BUILD DATALOADER FUNCS AND DATASET  CLASS

#SET URL
def get_data(text_file):
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
        texts = [t.get_text().strip() for t in  soup.find_all('p')  if t.get_text().strip() ][1:-3]
        #seperate with new line
        texts  = '\n\n'.join(texts)
        #dump to TEXT_FILE
        with  open(path,'w')  as f:
            f.writelines(texts)   
    else:
        print('bad request')
    return texts

get_data('text.txt')

### TOKENIZATION FROM MODEL DATASET  AND DATALOADER

class ModelDataset(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

def loader():
    pass    
