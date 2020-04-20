# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd


device = torch.device('cuda:0')

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
    
def readLangs(lang1, lang2, reverse=False):
    file1 = open('eng-fra.txt',encoding="utf8")
    lines = file1.readlines()
    count = 0
    
    pairs = [[normalizeString(s) for s in l.split('AMARTYA HATUA')] for l in lines]
    
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 256

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    pairs_new = []
    for pair in pairs:
        if (len(pair) >= 2):
            if(len(pair[0])<=256 and len(pair[1])<=256 ):
                input_lang.addSentence(pair[0])
                output_lang.addSentence(pair[1])
                pairs_new.append(pair)



    return input_lang, output_lang, pairs_new


input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
encoder1.load_state_dict(torch.load('encoder1.dict'))
attn_decoder1.load_state_dict(torch.load('attn_decoder1.dict'))

n_layers = 20
block_dim = 256
gp_lambda = 10
latent_dim = 256
interval = 1000
batch_size = 1
n_critic = 5

def train(epoch):


    generator = Generator(n_layers, block_dim)
    generator.to(device)
    critic = Critic(n_layers, block_dim)
    critic.to(device)
    g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
    c_optimizer = optim.Adam(critic.parameters(), lr=1e-4)
    

    encoder1.eval()
    attn_decoder1.eval()
    #autoencoder.eval()
    generator.train()
    critic.train()
    c_train_loss = 0.
    g_train_loss = 0.
    g_batches = 0
    hidden_size = 256
    max_length = 100




    for i in range(len(pairs)):
        # train critic
        #print("value of i = ", i)
        pair = pairs[i]
        sentence = pair[0]
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder1.initHidden()
        encoder_outputs = torch.zeros(input_length, encoder1.hidden_size, device=device)
        
        decoder_output = torch.zeros(max_length, latent_dim, device=device)

        fc3 = nn.Linear(encoder1.hidden_size, latent_dim)
        fc3.to(device)
        with torch.no_grad():
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder1(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

        encoder_outputs = fc3(encoder_outputs)
        c_optimizer.zero_grad()
        noise = torch.from_numpy(np.random.normal(0, 1, (input_length, latent_dim))).float()
        noise = noise.to(device)    	
        z_fake = generator(noise)        
        z_fake.to(device)
        real_score = critic(encoder_outputs)
        fake_score = critic(z_fake)
        grad_penalty = compute_grad_penalty(critic, encoder_outputs.data, z_fake.data)
        c_loss = -torch.mean(real_score) + torch.mean(fake_score) + gp_lambda * grad_penalty
        c_train_loss += c_loss.item()
        c_loss.backward()
        c_optimizer.step()



        # train generator
        if i % n_critic == 0:
            g_batches += 1
            g_optimizer.zero_grad()
            fake_score = critic(generator(noise))
            g_loss = -torch.mean(fake_score)
            g_train_loss += g_loss.item()
            g_loss.backward()
            g_optimizer.step()

        if interval > 0 and i % interval == 0:
            print('Epoch: {} | Batch: {}/{} ({:.0f}%) | G Loss: {:.6f} | C Loss: {:.6f}'.format(
                epoch, batch_size * i, len(pairs),
                       100. * (batch_size * i) / len(pairs),
                g_loss.item(), c_loss.item()
            ))

    print("End of loop ====>>>>>")
    g_train_loss /= g_batches
    c_train_loss /= len(pairs)
    print('* (Train) Epoch: {} | G Loss: {:.4f} | C Loss: {:.4f}'.format(
        epoch, g_train_loss, c_train_loss
    ))
    return (g_train_loss, c_train_loss)

def compute_grad_penalty(critic, real_data, fake_data):
    B = real_data.size(0)
    alpha = torch.FloatTensor(np.random.random((B, 1)))
    alpha = alpha.to(device)
    sample = alpha*real_data + (1-alpha)*fake_data
    sample.requires_grad_(True)
    sample = sample.to(device)
    score = critic(sample)
    outputs = torch.FloatTensor(B, 256).fill_(1.0)
    outputs.requires_grad_(False)
    outputs = outputs.to(device)
 
    grads = autograd.grad(
        outputs=score,
        inputs=sample,
        grad_outputs=outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    #grads = grads.view(B, -1)
    grad_penalty = ((grads.norm(2, dim=1) - 1.) ** 2).mean()

    return grad_penalty





generator = Generator(n_layers, block_dim)
critic = Critic(n_layers, block_dim)
critic.to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=1e-4)
c_optimizer = optim.Adam(critic.parameters(), lr=1e-4)

best_loss = np.inf
epochs = 2
for epoch in range(1, epochs + 1):
    g_loss, c_loss = train(epoch)
    loss = g_loss + c_loss
    if loss < best_loss:
        best_loss = loss
        print('* Saved')
        torch.save(generator.state_dict(), 'generator.th')
        torch.save(critic.state_dict(), 'critic.th')
