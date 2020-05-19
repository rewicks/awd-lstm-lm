import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import json

import data
from model import RNNModel

import os
import hashlib

import warnings
warnings.filterwarnings('ignore')

from utils import batchify, get_batch, repackage_hidden

import logging

logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)

processed_corpora_path = '/exp/rwicks/ersatz/data/corpora/'
###############################################################################
# Load data
###############################################################################

def model_save(prefix, model, criterion, optimizer):
    state_dict = model.state_dict()
    for layer in range(len(model.rnns)):
        bad_key = f'rnns.{layer}.module.weight_hh_l0'
        if bad_key in state_dict.keys():
            del state_dict[bad_key]
    with open(prefix+'.model', 'wb') as f:
        torch.save([state_dict, criterion, optimizer], f)
    with open(prefix+'.opt', 'w') as f:
        json.dump(model.opt, f)

def model_load(fn):
    model_opt = json.load(open(fn + '.opt'))
    model = RNNModel(model_opt['rnn_type'], model_opt['ntoken'], model_opt['ninp'], model_opt['nhid'], model_opt['nlayers'], model_opt['dropout'], model_opt['dropouth'], model_opt['dropouti'], model_opt['dropoute'], model_opt['wdrop'], model_opt['tie_weights'])
    with open(fn+'.model', 'rb') as f:
        model_dict, criterion, optimizer = torch.load(f)
        model.load_state_dict(model_dict) 
    return model, criterion, optimizer

def get_dataset(args, processed_corpora_path):
    #fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
    if args.train_path is not None:
        fn = os.path.join(processed_corpora_path + args.train_path.split('/')[-1])
        fn += '.train'
    elif args.test_path is not None:
        fn = os.path.join(processed_corpora_path + args.test_path.split('/')[-1])
    if args.valid_path is not None:
        fn += '.valid'
    if args.test_path is not None:
        fn += '.test'
    #fn = args.save + '.corpus'
    if os.path.exists(fn):
        logging.info('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        logging.info('Producing dataset...')
        corpus = data.Corpus(args.train_path, args.valid_path, args.test_path, vocab_file=args.vocab_path)
        torch.save(corpus, fn)
    return corpus

def batch_data(args, corpus):
    eval_batch_size = 10
    test_batch_size = 1
    train_data = None
    val_data = None
    test_data = None
    if args.train_path is not None:
        train_data = batchify(corpus.train, args.batch_size, args)
    if args.valid_path is not None:
        val_data = batchify(corpus.valid, eval_batch_size, args)
    if args.test_path is not None:
        test_data = batchify(corpus.test, test_batch_size, args)

    return train_data, val_data, test_data

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss

def build_model(args, corpus):
    criterion = None
    ntokens = len(corpus.dictionary)
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    ###
    if args.resume:
        logging.info('Resuming model ...')
        model, criterion, optimizer = model_load(args.resume_path)
        optimizer.param_groups[0]['lr'] = args.lr
        model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
        if args.wdrop:
            from weight_drop import WeightDrop
            for rnn in model.rnns:
                if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
                elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
    ###
    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        logging.info(f'Using {splits}')
        criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
    ###
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    logging.info(f'Args: {args}')
    logging.info(f'Model total parameters: {total_params}')

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    return model, criterion

###############################################################################
# Training code
###############################################################################


def evaluate(model, criterion, data_source, ntokens, batch_size=10, dividing_constant=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    #ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)

        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    if dividing_constant is None:
        return total_loss.item() / len(data_source)
    else:
        return total_loss.item() / (dividing_constant/batch_size)


def train(args, ntokens, train_data, model, criterion, optimizer, params, epoch):
    # Turn on training mode which enables dropout.

    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    #ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            logging.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))

            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

def run_epoch(args, train_data, ntokens, val_data, model, criterion):
    # Loop over epochs.
    lr = args.lr
    best_val_loss = []
    stored_loss = 100000000

    params = list(model.parameters()) + list(criterion.parameters())

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(args, ntokens, train_data, model, criterion, optimizer, params, epoch)
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    if 'ax' in optimizer.state[prm]:
                        prm.data = optimizer.state[prm]['ax'].clone()
                        

                val_loss2 = evaluate(model, criterion, val_data, ntokens, dividing_constant=70390)
                logging.info('-' * 89)
                logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:.4f} | '
                    'valid ppl {:.6f} | valid bpc {:8.3f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                logging.info('-' * 89)


                if val_loss2 < stored_loss:
                    model_save(args.save, model, criterion, optimizer)
                    logging.info('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    if prm in tmp.keys():
                        prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate(model, criterion, val_data, ntokens, dividing_constant=70390)
                logging.info('-' * 89)
                logging.info('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:.6f} | '
                    'valid ppl {:.6f} | valid bpc {:8.3f}'.format(
                  epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                logging.info('-' * 89)


                if val_loss < stored_loss:
                    model_save(args.save, model, criterion, optimizer)
                    logging.info('Saving model (new best validation)')
                    stored_loss = val_loss

                #if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                #    logging.info('Switching to ASGD')
                    #optimizer = torch.optim.ASGD(params, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                #    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                if epoch in args.when:
                    logging.info('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(args.save, epoch), model, criterion, optimizer)
                    logging.info('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        logging.info('-' * 89)
        logging.info('Exiting from training early')


def test(args, test_data, test_batch_size=1):
    # Load the best saved model.
    model, criterion, _ = model_load(args.save)

    if torch.cuda.is_available() and args.cuda:
        model.cuda()

    # Run on test data.
    test_loss = evaluate(model, criterion, test_data, model.opt['ntoken'], test_batch_size)
    logging.info('=' * 89)
    logging.info('| end of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    logging.info('=' * 89)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')

    parser.add_argument('--train_path', type=str, default=None,
                        help='path to the training file. if none is provided, only testing will occur')
    parser.add_argument('--valid_path', type=str, default=None,
                        help='path to validation file. if none is provided, best model will be based off of training data')
    parser.add_argument('--test_path', type=str, default=None,
                        help='path to test file. if none is provided, only training will occur')

    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=400,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1150,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=30,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8000,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=70,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.3,
                        help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.65,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--nonmono', type=int, default=5,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    randomhash = ''.join(str(time.time()).split('.'))
    parser.add_argument('--save', type=str, default=randomhash + '.pt',
                        help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')
    parser.add_argument('--resume', action='store_true',
                        help='whether or not you should load a presaved model')
    #parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer to use (sgd, adam)')
    parser.add_argument('--when', nargs="+", type=int, default=[-1],
                        help='When (which epochs) to divide the learning rate by 10 - accepts multiple')

    parser.add_argument('--vocab_path', type=str, default=None,
                        help='Path to json vocab path')
    parser.add_argument('--log_path', type=str, default='train.log')

    args = parser.parse_args()
    args.tied = True

    if args.train_path == "None":
        args.train_path = None
    if args.valid_path == "None":
        args.valid_path = None
    if args.test_path == "None":
        args.test_path = None
    if args.vocab_path == "None":
        args.vocab_path = None

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            logging.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    corpus = get_dataset(args, processed_corpora_path)
    train_data, val_data, test_data = batch_data(args, corpus)

    args.when = [20, 40, 60]    

    if args.train_path is not None:
        model, criterion = build_model(args, corpus)
        run_epoch(args, train_data, len(corpus.dictionary), val_data, model, criterion)

    if args.test_path is not None:
        test(args, test_data)

