import nltk
import json
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk.stem.porter import PorterStemmer
import numpy as np
import random

nltk.download('punkt')

def tokenizer(sentence):
    return nltk.word_tokenize(sentence)

def stem(word,stemmer):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,stemmer,all_words):
    tokenized_sentence=[stem(w,stemmer)for w in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1.0

    return bag

class ChatDataset(Dataset):
    def __init__(self,x_train,y_train):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.n_samples

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()  # Call the superclass's __init__ method
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()  # Note the corrected capitalization here

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)  # Note the corrected method name here
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out



