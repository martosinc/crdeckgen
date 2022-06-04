#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn

cards = ['START', 'mini-pekka', 'ice-wizard', 'zap', 'baby-dragon', 'fisherman', 'night-witch', 'bomber', 'hunter', 'freeze', 'tombstone', 'lava-hound', 'inferno-tower', 'valkyrie', 'bandit', 'lightning', 'earthquake', 'mega-knight', 'electro-spirit', 'bats', 'fireball', 'spear-goblins', 'mega-minion', 'knight', 'golden-knight', 'ice-golem', 'zappies', 'firecracker', 'electro-wizard', 'rocket', 'clone', 'wall-breakers', 'miner', 'archer-queen', 'skeleton-army', 'musketeer', 'dart-goblin', 'fire-spirit', 'skeleton-king', 'goblin-barrel', 'goblin-gang', 'cannon-cart', 'giant-skeleton', 'battle-ram', 'x-bow', 'royal-giant', 'flying-machine', 'electro-dragon', 'archers', 'princess', 'tornado', 'elite-barbarians', 'the-log', 'balloon', 'goblin-drill', 'royal-hogs', 'three-musketeers', 'tesla', 'magic-archer', 'lumberjack', 'golem', 'inferno-dragon', 'pekka', 'skeletons', 'graveyard', 'skeleton-dragons', 'guards', 'skeleton-barrel', 'mortar', 'arrows', 'bowler', 'heal-spirit', 'royal-ghost', 'bomb-tower', 'hog-rider', 'barbarian-barrel', 'royal-delivery', 'poison', 'ice-spirit', 'barbarians', 'cannon', 'mighty-miner', 'elixir-collector']

n_cards = len(cards)

card_to_int = dict([(b,a) for a,b in enumerate(cards)])

class RNN(nn.Module):
    def __init__(self,input_size, hidden_size,output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        
    def forward(self,input,hidden):
        input_combined = torch.cat((input,hidden),1)
        hidden = torch.tanh(self.i2h(input_combined))
        output = self.h2o(hidden)
        output = nn.LogSoftmax(dim=1)(output)
        return output, hidden
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = RNN(n_cards, 128, n_cards)
rnn.load_state_dict(torch.load('crmodel.pth'))


def cardTensor(cd):
    tensor = torch.zeros(1,1,n_cards)
    tensor[0,0,card_to_int[cd]] = 1
    return tensor

max_length = 8

def sample(start_card='START'):
    with torch.no_grad():
        input = cardTensor(start_card)
        hidden = rnn.initHidden()
        
        output_deck = []
        
        for i in range(max_length):
            output, hidden = rnn(input[0], hidden)
            
            t = output[0]
            
            for ix in sorted(list(map(lambda x: cards.index(x), output_deck)))[::-1]:
                t = torch.cat([t[:ix],t[ix+1:]])
            
            c = torch.distributions.categorical.Categorical(logits=t)
            q = c.sample()
            i = list(output[0]).index(t[q])
            
            card = cards[i]

            output_deck.append(card)

        return output_deck

def LinkGen(deck):
    url = 'https://www.deckshop.pro/check/?deck='
    for card in deck:
        card = card.replace('-','')
        url += card + '-'
    return url[:-1]

for i in range(50):
    decksample = sample()
    
    print(decksample)
    print(LinkGen(decksample))