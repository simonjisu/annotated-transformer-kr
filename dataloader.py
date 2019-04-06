import spacy
import torch
import torchtext.datasets as datasets
from torchtext import datasets
from torchtext.data import Field, BucketIterator, TabularDataset


def get_data(args):
    # batch
    batch_size = args.batch
    device = "cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu"
    if args.data_type == "multi30k":
        
        spacy_de = spacy.load('de')
        spacy_en = spacy.load('en')

        def tokenize_de(text):
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            return [tok.text for tok in spacy_en.tokenizer(text)]

        # set up fields
        src = Field(tokenize=tokenize_en, 
                    use_vocab=True, 
                    lower=True, 
                    include_lengths=False, 
                    fix_length=args.max_length, # fix max length
                    batch_first=True)
        trg = Field(tokenize=tokenize_de, 
                    use_vocab=True,
                    init_token='<s>',
                    eos_token='</s>', 
                    lower=True, 
                    fix_length=args.max_length,  # fix max length
                    batch_first=True)

        # make splits for data
        train, valid, test = datasets.Multi30k.splits(('.en', '.de'), 
                                                      (src, trg), 
                                                      root=args.root_dir)

        # build the vocabulary
        src.build_vocab(train.src, min_freq=args.min_freq)
        trg.build_vocab(train.trg, min_freq=args.min_freq)
        
        # make iterator for splits
        train_iter, valid_iter, test_iter = BucketIterator.splits((train, valid, test), batch_sizes=([batch_size]*3), device=device)
        
        return (src, trg), (train, valid, test), (train_iter, valid_iter, test_iter)