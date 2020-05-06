import os
import torch
import json

from collections import Counter


class Dictionary(object):
    def __init__(self, vocab_path=None):
        if vocab_path is not None:
            with open(vocab_path) as vocabfile:
                self.word2idx = json.load(vocabfile)
            self.idx2word = ['' for x in range(len(self.word2idx))]
            for word in self.word2idx:
                self.idx2word[self.word2idx[word]] = word
            self.total = 0
            self.counter = Counter() 
        else:
            self.word2idx = {}
            self.idx2word = []
            self.counter = Counter()
            self.total = 0

    def add_word(self, word, preexist=True):
        if word not in self.word2idx and not preexist:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx.get(word, self.word2idx['<unk>'])
        self.counter[token_id] += 1
        self.total += 1
        return token_id

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, vocab_file=None):
        self.dictionary = Dictionary(vocab_file)
        if vocab_file is not None:
            preexist=True
        else:
            preexist=False
        self.train = self.tokenize(os.path.join(path, 'train.txt'), preexist=preexist)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), preexist=preexist)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), preexist=preexist)

    def tokenize(self, path, preexist=True):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['</s>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word, preexist=preexist)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['</s>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
