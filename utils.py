import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtext
from torchtext.datasets import LanguageModelingDataset
from mosestokenizer import *
import gensim
from tqdm import tqdm

BS = 128 # batch_size
EMBED_SIZE = 100 # embedding layer size
stop_words = ['&quot;', '«', '»', '–', '„', '“', '@', '@-@', '—', '-',
              '”', '_', '~', '`', '\xad'] + list('"#$%&\'()*+, -/:;<=>@[\]^_`{|}~')

NROWS = 300000 # number of rows to read

class NgramModel(object):

    def __init__(self, n):
        self.n = n
        # dictionary where (key: context, value: dict (with key: target_word, value: count))
        self.context_to_target_counts = {}
        
        # dictionary which translates word to index
        self.word_to_index = {'<unk>' : 0}

        # dictionary which translates index to word
        self.index_to_word = {0 : '<unk>'}
        self.word_index_counter = 1

        # Smoothing coefficient to not have 0 probability problem
        self.SMOOTHING = 0.001
    
    def index_word(self, word):
        """
            checks if already had seen this word.
            if it's new word, assigns index to it and adds to dictionaries.
        """
        if word not in self.word_to_index:
            self.word_to_index[word] = self.word_index_counter
            self.index_to_word[self.word_index_counter] = word
            self.word_index_counter += 1
    
    def get_index_from_word(self, word):
        """return index of word. if there is no such word, return index of <unk> token"""
        if word not in self.word_to_index:
            word = '<unk>'

        return self.word_to_index[word]
    
    def get_word_from_index(self, index):
        "returns word for given index"
        return self.index_to_word[index]

    def update(self, tokens):
        """updates whole state dictionaries, using sliding window algorithm"""
        tokens.append('<eos>')
        context = (self.n - 1) * ['<sos>']
        
        for target in tokens:
            self.index_word(target)

            context_tuple = tuple(context)
            if context_tuple in self.context_to_target_counts:
                if target in self.context_to_target_counts[context_tuple]:
                    self.context_to_target_counts[context_tuple][target] += 1
                else:
                    self.context_to_target_counts[context_tuple][target] = 1
            else:
                self.context_to_target_counts[context_tuple] = {target : 1}
            
            context.pop(0)
            context.append(target)
        
    def uniform_distribution(self):
        """return uniform distribution on whole vocab"""
        vocab_size = len(self.word_to_index)
        return torch.FloatTensor(vocab_size).fill_(1 / vocab_size)

    def get_distribution(self, context):
        """return distribution on whole vocab depend on context"""
        context = context.copy()
        while len(context) < self.n - 1:
            context.insert(0, '<sos>')
        context = context[:self.n - 1]

        context_tuple = tuple(context)
        if context_tuple not in self.context_to_target_counts:
            return self.uniform_distribution()

        vocab_size = len(self.word_to_index)
        targets_dict = self.context_to_target_counts[context_tuple]

        target_counts_sum = 0
        for _, count in targets_dict.items():
            target_counts_sum += count
        
        result = torch.FloatTensor(vocab_size)
        for word in self.word_to_index:
            word_index = self.get_index_from_word(word)
            if word in targets_dict:
                count = targets_dict[word]
                result[word_index] = (count + self.SMOOTHING) /  (target_counts_sum + vocab_size * self.SMOOTHING)
            else:
                result[word_index] = (self.SMOOTHING) / (target_counts_sum + vocab_size * self.SMOOTHING)

        return result
    

def n_gram_compute_perplexity(model, n, paragraphs, tokenizer):
    loss = 0
    counter = 0
    for paragraph in tqdm(paragraphs):
        tokens = tokenizer(paragraph)
        tokens.append('<eos>')
        context = n * ['<sos>']
        for target in tokens:
            counter += 1
            y_pred = model.get_distribution(context)
            y = model.get_index_from_word(target)
            
            loss += torch.nn.functional.cross_entropy(torch.unsqueeze(y_pred, 0), torch.tensor([y])).item()

            context.pop(0)
            context.append(target)
    
    return np.exp(loss / counter)


class TextField():
    """Class responsible for reading data, spliting into train, validation and test sets, building vocabulary and data sets"""
    def __init__(self, w2v_model_path, df_path,batch_first=True):
        w2v_model = gensim.models.Word2Vec.load(w2v_model_path)
        tokenizer = MosesTokenizer()
        self._txt_field =  torchtext.data.Field(
            sequential=True, # we tell the Torchtext that our data in sequential (sequence of words)
            use_vocab=True, # we need Torchtext to build a vocabulary for us (as opposed of building it manually, maybe from another data source)
            lower=False, # we dont need to lower sentences
            tokenize=tokenizer, 
            batch_first=batch_first,
            eos_token='<eos>', # add end of sentence token
            stop_words=stop_words # stop words
        )
        data_df = pd.read_csv(df_path, lineterminator='\n', nrows=NROWS) 
        # split data into train, validation and test sets
        self.train_df, self.validate_df, self.test_df = np.split(data_df.sample( frac=1, random_state=42), [int(.70*len(data_df)), int(.85*len(data_df))])
        self.train_df.to_csv('train.csv', index=False, header=False)
        self.validate_df.to_csv('valid.csv', index=False, header=False)
        self.test_df.to_csv('test.csv', index=False, header=False)
        # create data sets for data frames
        self._train_ds, self._valid_ds, self._test_ds = LanguageModelingDataset.splits(
            path='',
            train='train.csv',
            validation='valid.csv',
            test='test.csv',
            newline_eos=True,
            text_field=self._txt_field
        )
        # build txt_field vocabulary
        self._txt_field.build_vocab(
            # We only use training set (not train and validation together) so that we could measure model performance on OOV (out-of-vocabulary) words in validation set.
            self._train_ds,

            # Dropping infrequent words is usually a good idea, for example, to get rid of misspelings in informal text.
            # Besides, model won't learn anything good from words which are met only few times in entire corpus.
            min_freq=5,
        )
        # load pretrained embeddings into torchtext field
        word2vec_vectors = []
        for token, idx in self._txt_field.vocab.stoi.items():
            if token in w2v_model.wv.vocab.keys():
                word2vec_vectors.append(torch.FloatTensor(w2v_model[token]))
            else:
                word2vec_vectors.append(torch.zeros(EMBED_SIZE))
        self._txt_field.vocab.set_vectors(self._txt_field.vocab.stoi, word2vec_vectors, EMBED_SIZE)
    
    
    def get_txt_field(self):
        """return torchtext field"""
        return self._txt_field

    def get_data_sets(self):
        """return data sets"""
        return self._train_ds, self._valid_ds, self._test_ds


class DataLoader():
    """Class responsible for creating train, validation and test data loaders from torchtext field"""
    def __init__(self, text_field, device):
        train_ds, valid_ds, test_ds = text_field.get_data_sets()
        # create data loaders using torchtext.data.BPTTIterator
        self._train_dl = torchtext.data.BPTTIterator(
            dataset=train_ds,
            batch_size=BS, 
            sort_key=lambda x: len(x.text),  # minimize padddings
            bptt_len=10, # maximum length of a sequence in a batch (bptt means Backpropagation through time, which, in current context, has nothing to do with BPTT algorithm).
            train=True, # Indicator to specify that this is training set (Data shuffling between epochs is done during training, not during testing).
            device=device)  # On which device to pre-load batch tensors.

        self._dev_dl = torchtext.data.BPTTIterator(
            dataset=valid_ds,
            batch_size=4 * BS, # we can make dev batch size bigger, since we don't do backprop and save memory.
            sort_key=lambda x: len(x.text),
            bptt_len=10,
            train=True,
            device=device)

        self._test_dl = torchtext.data.BPTTIterator(
            dataset=valid_ds, # we can make dev batch size bigger, since we don't do backprop and save memory.
            batch_size=4 * BS,
            sort_key=lambda x: len(x.text),
            bptt_len=10,
            train=True,
            device=device)

    def get_dls(self):
        """return data loaders"""
        return self._train_dl, self._dev_dl, self._test_dl



def top_p(probs, p=0.92):
    """ 
    First we choose 1000 words with most probability.
    Then we choose smallest possible set of words whose cumulative probability exceeds the probability p, if no such set exists we choose all 1000 of them.
    Choose random word in this set with respect of their probabilities.
    """
    top_k = torch.topk(probs, dim=-1, k=1000) # get top 1000 words
    indices = top_k.indices # their indices
    sorted = top_k.values # their values
    cumulative_probs = torch.cumsum(sorted, dim=0) # calculate cumulative sum
    
    # choose first index where cumulative sum is greater than p, if no such index choose last index
    cut_index = len(cumulative_probs) - 1
    for i , x in enumerate(cumulative_probs):
        if x >= p and i >= 5:
            cut_index = i
            break
    # choose subset and normilize their probability so sum would be 1
    probs = sorted[:cut_index+1]
    probs = probs.cpu().numpy()
    probs /= sum(probs)

    indices = indices[:cut_index+1]
    
    return np.random.choice(indices.cpu(), 1, replace=False, p=probs)[0] # return random word with respect of their probabilities



class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, txt_field, device, num_layers=1):

        super().__init__()

        self.hidden_dim = hidden_dim # hidden_dim 
        self.num_layers = num_layers # number of layers in lstm model
        self.device = device # device 
    
        self.emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(txt_field.vocab.vectors), # create embeding layer with pretrained weights
            freeze=False # set freeze to False so it would learn embedding layer in the process (tuning)
        ) 

        self.lstm = nn.LSTM(
                    input_size=embedding_dim, # input_size 
                    hidden_size=hidden_dim, # hidden dimension size
                    bias=True, # we need bias
                    batch_first=True, # we train our model with batch_first
                    bidirectional=True, # lstm needs to be biderectional, which gives better results
                    num_layers=self.num_layers # number of layers in lstm
                    ).to(self.device)
        self.dropout = nn.Dropout(0.3) # add dropout to prevent neural network from overfitting.
        self.classifier = nn.Linear(2*hidden_dim, vocab_size).to(self.device) # linear layer
        self.vocab_size = vocab_size # vocab_size

    def forward(self, inp):
        # Step 1: 
        # Word indices need to be transformed into embedding vectors.
        # inp tensor of shape (batch_size, max_seq_len) must be passed through embedding layer and output will be (batch_size, max_seq_len, embedding_dim) tensor.
    
        inp = self.emb(inp).to(self.device)
        
        # Step 2:
        # dropout layer 
        # inp tensor of shape (batch_size, max_seq_len, embedding_dim) must be passed through dropout and output will be (batch_size, max_seq_len, embedding_dim) tensor.
        inp = self.dropout(inp)
        
        # Step 3: 
        # Pass embedding sequence into lstm and get all hidden states in a tensor (batch_size, max_seq_len, 2 * hidden_size)
        # last dimension is 2 * hidden_size because lstm is bidirectional
        out, (hn,cn) = self.lstm(inp)
        
        # Step 4:
        # Pass each hidden state into classifier to get probability distributions over the vocabulary (We won't apply softmax here, but during training below).
        # I.e. each hidden state will be used to make prediction - current word, given previous words.
        # To avoid looping over hidden states and individually passing them into classifier, we gonna reshape hs tensor 
        # from (batch_size, max_seq_len, 2 * hidden_size) into (batch_size * max_seq_len, 2 * hidden_size), pass it through classifier, reshape back!
        # result must be (batch_size, max_seq_len, vocab_size) tensor. Last dimension is probability distribution over words. 
        out = out.reshape(-1, 2*self.hidden_dim)
        
        return self.classifier(out).view(inp.size(0), inp.size(1), -1).to(self.device)
        
        
        

def compute_perplexity(model, dl, device):
    """
    Compute perplexity
    """
    model.eval()
    with torch.no_grad(): # tells Pytorch not to store values of intermediate computations for backward pass because we not gonna need gradients.
        loss = 0
        for batch in dl:
            x, y = batch.text.to(device), batch.target.to(device)
            y_pred = model(x)
            loss += torch.nn.functional.cross_entropy(y_pred.view(x.size(0)*x.size(1), -1), y.view(-1)).item()
    
    model.train()
    return np.exp(loss / len(dl))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_loop(model, train_dl, dev_dl, device, epochs, model_save_path):

    model.train()

    # we add weight decay (L2 regularization) to avoid overfitting.
    # try removing it and see what happens.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # we will reduce initial learning rate by 'lr=lr*factor' every time validation perplexity doesn't improve within certain range.
    # details here https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau 
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6, patience=10) 

    crit = nn.CrossEntropyLoss(reduction='mean')

    it = 1
    total_loss = 0
    curr_perplexity = None
    perplexity = None

    for epoch in range(epochs):
        for i, batch in enumerate(train_dl):
            # if we don't do this, pytorch will accumulate gradients (summation) from previous backprop to next backprop and so on...
            # this behaviour is useful when we can't fit many samples in memory but we need to compute gradient on large batch. In this case, we simply accumulate gradient and do backprop
            # after enough samples has been seen.
            optimizer.zero_grad()

            x, y = batch.text.to(device), batch.target.to(device)

            # do forward pass, will save intermediate computations of the graph for later backprop use.
            y_pred = model(x)

            # y_pred has shape (batch_size, max_seq_len, vocab_size), y has shape (batch_size, max_seq_len), and we 
            # need to compute average Cross Entropy across batch and sequence dimensions. For this, we first reshape tensors accordingly. 
            loss = crit(y_pred.view(x.size(0)*x.size(1), -1), y.view(-1))
            
            total_loss += loss.item()
            
            # running backprop.
            loss.backward()

            # doing gradient descent step.
            optimizer.step()

            # we are logging current loss/perplexity in every 100 iteration (in every 100 batch)
            if it % 100 == 0:
                # computing validation set perplexity in every 500 iteration.
                if it % 500 == 0:
                    curr_perplexity = compute_perplexity(model, dev_dl, device)

                    lr_scheduler.step(curr_perplexity)

                    # making checkpoint of best model weights.
                    if not perplexity or curr_perplexity < perplexity:
                        torch.save(model.state_dict(), model_save_path)
                        perplexity = curr_perplexity

                print('Epoch', epoch + 1, '| Iter', it, '| Avg Train Loss', total_loss / 100, '| Dev Perplexity', curr_perplexity, '| LR ', get_lr(optimizer))
                total_loss = 0

            it += 1
