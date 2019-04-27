import spacy
import torch
import torchtext.datasets as datasets
from torchtext import datasets
from torchtext.data import Field, BucketIterator, TabularDataset

class SplitReversibleField(Field):
    """ref: http://anie.me/On-Torchtext/"""
    def __init__(self, **kwargs):
        if kwargs.get('tokenize') is list:
            self.use_revtok = False
        else:
            self.use_revtok = True
        if kwargs.get('tokenize') not in ('revtok', 'subword', list):
            kwargs['tokenize'] = 'revtok'
        if 'unk_token' not in kwargs:
            kwargs['unk_token'] = ' <unk> '
        super(SplitReversibleField, self).__init__(**kwargs)

    def reverse(self, batch):
        if self.use_revtok:
            try:
                import revtok
            except ImportError:
                print("Please install revtok.")
                raise
        if not self.batch_first:
            batch = batch.t()
        with torch.cuda.device_of(batch):
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        if self.use_revtok:
            return [revtok.detokenize(ex) for ex in batch]
        return [' '.join(ex) for ex in batch]

def get_data(args):
    # batch
    batch_size = args.batch
    device = "cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu"
    
        
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # set up fields
    src = SplitReversibleField(tokenize=tokenize_en, 
                use_vocab=True, 
                lower=True, 
                include_lengths=False, 
                fix_length=args.max_length, # fix max length
                batch_first=True)
    trg = SplitReversibleField(tokenize=tokenize_de, 
                use_vocab=True,
                init_token='<s>',
                eos_token='</s>', 
                lower=True, 
                fix_length=args.max_length,  # fix max length
                batch_first=True)
    
    if args.data_type == "multi30k":
        # make splits for data
        train, valid, test = datasets.Multi30k.splits(('.en', '.de'), 
                                                      (src, trg), 
                                                      root=args.root_dir)
        # build the vocabulary
        src.build_vocab(train.src, min_freq=args.min_freq)
        trg.build_vocab(train.trg, min_freq=args.min_freq)
        
        
        
    elif args.data_type == "wmt14":
        # make splits for data
        train, valid, test = datasets.WMT14.splits(('.en', '.de'), 
                                                   (src, trg), 
                                                   root=args.root_dir)
        # build the vocabulary
        src.build_vocab(train.src, min_freq=args.min_freq)
        trg.build_vocab(train.trg, min_freq=args.min_freq)
        
    elif args.data_type == "iswlt":
        # make splits for data
        train, valid, test = datasets.IWSLT.splits(('.en', '.de'), 
                                                   (src, trg), 
                                                   root=args.root_dir)
        # build the vocabulary
        src.build_vocab(train.src, min_freq=args.min_freq)
        trg.build_vocab(train.trg, min_freq=args.min_freq)
        
    else:
        assert False, "Please Insert data_type"

    # make iterator for splits
    train_iter, valid_iter, test_iter = BucketIterator.splits((train, valid, test), batch_sizes=([batch_size]*3), device=device)
        
    return (src, trg), (train, valid, test), (train_iter, valid_iter, test_iter)