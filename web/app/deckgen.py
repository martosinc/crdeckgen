import torch
import zipfile

class RNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size,output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size+hidden_size, hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self,input,hidden):
        input_combined = torch.cat((input,hidden),1)
        hidden = torch.tanh(self.i2h(input_combined))
        output = self.h2o(hidden)
        output = torch.nn.LogSoftmax(dim=1)(output)
        return output, hidden
    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class Deckgen():
    def __init__(self):
        self.cards = ['START', 'mini-pekka', 'ice-wizard', 'zap', 'baby-dragon', 'fisherman', 'night-witch', 'bomber', 'hunter', 'freeze', 'tombstone', 'lava-hound', 'inferno-tower', 'valkyrie', 'bandit', 'lightning', 'earthquake', 'mega-knight', 'electro-spirit', 'bats', 'fireball', 'spear-goblins', 'mega-minion', 'knight', 'golden-knight', 'ice-golem', 'zappies', 'firecracker', 'electro-wizard', 'rocket', 'clone', 'wall-breakers', 'miner', 'archer-queen', 'skeleton-army', 'musketeer', 'dart-goblin', 'fire-spirit', 'skeleton-king', 'goblin-barrel', 'goblin-gang', 'cannon-cart', 'giant-skeleton', 'battle-ram', 'x-bow', 'royal-giant', 'flying-machine', 'electro-dragon', 'archers', 'princess', 'tornado', 'elite-barbarians', 'the-log', 'balloon', 'goblin-drill', 'royal-hogs', 'three-musketeers', 'tesla', 'magic-archer', 'lumberjack', 'golem', 'inferno-dragon', 'pekka', 'skeletons', 'graveyard', 'skeleton-dragons', 'guards', 'skeleton-barrel', 'mortar', 'arrows', 'bowler', 'heal-spirit', 'royal-ghost', 'bomb-tower', 'hog-rider', 'barbarian-barrel', 'royal-delivery', 'poison', 'ice-spirit', 'barbarians', 'cannon', 'mighty-miner', 'elixir-collector']

        self.n_cards = len(self.cards)

        self.card_to_int = dict([(b,a) for a,b in enumerate(self.cards)])

        self.rnn = RNN(self.n_cards,128,self.n_cards)
        with zipfile.ZipFile("app/crmodel.zip", 'r') as zip_ref:
            # zip_ref.extractall(".")
            # zip_ref.
            # self.rnn.load_state_dict(torch.load('app/crmodel.pth'))
            self.rnn.load_state_dict(torch.load(zip_ref.open('crmodel.pth')))
    
    def cardTensor(self,cd):
        tensor = torch.zeros(1,1,self.n_cards)
        tensor[0,0,self.card_to_int[cd]] = 1
        return tensor
    
    def sample(self,start_card='START'):
        with torch.no_grad():
            input = self.cardTensor(start_card)
            hidden = self.rnn.initHidden()
            
            output_deck = []
            
            for i in range(8):
                output, hidden = self.rnn(input[0], hidden)
                
                t = output[0]
                
                for ix in sorted(list(map(lambda x: self.cards.index(x), output_deck)))[::-1]:
                    t = torch.cat([t[:ix],t[ix+1:]])
                
                c = torch.distributions.categorical.Categorical(logits=t)
                q = c.sample()
                i = list(output[0]).index(t[q])
                
                card = self.cards[i]

                output_deck.append(card)

            return self.LinkGen(output_deck)
    
    def LinkGen(self, deck):
        url = 'https://www.deckshop.pro/check/?deck='
        for card in deck:
            card = card.replace('-','')
            url += card + '-'
        return url[:-1]
