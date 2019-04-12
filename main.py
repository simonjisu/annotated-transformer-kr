import argparse
import torch
import torch.optim as optim
from dataloader import get_data
from transformer.models import Transformer
from transformer.labelsmooth import LabelSmoothing
from transformer.warmupoptim import WarmUpOptim
from trainer import Trainer


def argument_parsing(preparse=False):
    parser = argparse.ArgumentParser(description="Transformer Argparser",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # load data
    parser.add_argument("-rt", "--root_dir", required=True,
                   help="Root Dir")
    parser.add_argument("-dt", "--data_type", type=str, default="multi30k",
                   help="Dataset type: wmt14, multi30k, iwslt")
    parser.add_argument("-maxlen", "--max_length", type=int, default=None,
                   help="Max length of Sentences")
    parser.add_argument("-minfreq", "--min_freq", type=int, default=1,
                   help="Minmum frequence of vocabulary")

    # model
    parser.add_argument("-nl", "--n_layer", type=int, default=6,
                   help="Number of layers in Encoder / Decoder")
    parser.add_argument("-nh", "--n_head", type=int, default=8,
                   help="Number of heads in Multi-head Attention sublayer")
    parser.add_argument("-dm", "--d_model", type=int, default=512,
                   help="Dimension of model")
    parser.add_argument("-dk", "--d_k", type=int, default=64,
                   help="Dimension of key")
    parser.add_argument("-dv", "--d_v", type=int, default=64,
                   help="Dimension of value")
    parser.add_argument("-df", "--d_f", type=int, default=2048,
                   help="Dimension of value")
    parser.add_argument("-pad", "--pad_idx", type=int,
                   help="Pad index of vocabulary")
    parser.add_argument("-pospad", "--pos_pad_idx", type=int,
                   help="Position pad index")
    parser.add_argument("-drop", "--drop_rate", type=float, default=0.1,
                   help="Drop Rate")
    parser.add_argument("-lws","--linear_weight_share", action="store_true",
                   help="Share the same weight matrix between the decoder embedding layer and the pre-softmax linear transformation")
    parser.add_argument("-ews","--embed_weight_share", action="store_true",
                   help="Share the same weight matrix between the decoder embedding layer and the encoder embedding layer")
    parser.add_argument("-conv","--use_conv", action="store_true",
                   help="Use Convolution operation in PositionWiseFFN layer")

    # loss function
    parser.add_argument("-eps", "--smooth_eps", type=float, default=0.1,
                   help="Label smoothing epsilon value")

    # optimizer
    parser.add_argument("-warm", "--warmup_steps", type=int, default=4000,
                   help="Warmup steps for learning rate schedule")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9,
                   help="Beta1 value for Adam optimizer")
    parser.add_argument("-b2", "--beta2", type=float, default=0.98,
                   help="Beta2 value for Adam optimizer")

    # training
    parser.add_argument("-bt","--batch", type=int, default=64,
                   help="Mini batch size")
    parser.add_argument("-step","--n_step", type=int, default=30,
                   help="Total Training Step")
    parser.add_argument("-cuda","--use_cuda", action="store_true",
                   help="Use Cuda")
    parser.add_argument("-svp","--save_path", type=str, default="./saved_model/model.pt",
                   help="Path to save model")
    parser.add_argument("-load","--load_path", type=str,
                   help="load previous model to transfer learning")
    parser.add_argument("-vb","--verbose", type=int, default=0,
                   help="verbose")
    parser.add_argument("-met","--metrics_method", type=str, default="acc",
                   help="verbose")
    
    
    if preparse:
        return parser
    
    args = parser.parse_args()
    return args

def main(args):
    # configs path to save model
    from pathlib import Path
    p = Path(args.save_path).parent
    if not p.exists():
        p.mkdir()
    
    
    device = "cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu"
    import sys
    print(sys.version)
    print(f"Using {device}")
    print("Loading Data...")
    (src, trg), (train, valid, _), (train_loader, valid_loader, _) = get_data(args)
    src_vocab_len = len(src.vocab.stoi)
    trg_vocab_len = len(trg.vocab.stoi)
    print(f"SRC vocab {src_vocab_len}, TRG vocab {trg_vocab_len}")
    enc_max_seq_len = args.max_length
    dec_max_seq_len = args.max_length
    pad_idx = src.vocab.stoi["<pad>"] if args.pad_idx is None else args.pad_idx
    pos_pad_idx = 0 if args.pos_pad_idx is None else args.pos_pad_idx
    
    print("Building Model...")
    model = Transformer(enc_vocab_len=src_vocab_len, 
                        enc_max_seq_len=enc_max_seq_len, 
                        dec_vocab_len=trg_vocab_len, 
                        dec_max_seq_len=dec_max_seq_len, 
                        n_layer=args.n_layer, 
                        n_head=args.n_head, 
                        d_model=args.d_model, 
                        d_k=args.d_k, 
                        d_v=args.d_v, 
                        d_f=args.d_f, 
                        pad_idx=pad_idx, 
                        pos_pad_idx=pos_pad_idx, 
                        drop_rate=args.drop_rate, 
                        use_conv=args.use_conv, 
                        linear_weight_share=args.linear_weight_share, 
                        embed_weight_share=args.embed_weight_share).to(device)
    
    if args.load_path is not None:
        print(f"Load Model {args.load_path}")
        model.load_state_dict(torch.load(args.load_path))
    
    loss_function = LabelSmoothing(trg_vocab_size=trg_vocab_len, 
                                   pad_idx=args.pad_idx, 
                                   eps=args.smooth_eps)
    
    optimizer = WarmUpOptim(warmup_steps=args.warmup_steps, 
                            d_model=args.d_model, 
                            optimizer=optim.Adam(model.parameters(),
                                             betas=(args.beta1, args.beta2), 
                                             eps=10e-9))
    
    trainer = Trainer(optimizer=optimizer, 
                      train_loader=train_loader, 
                      test_loader=valid_loader, 
                      n_step=args.n_step, 
                      device=device, 
                      save_path=args.save_path,
                      metrics_method=args.metrics_method,
                      verbose=args.verbose)
    print("Start Training...")
    trainer.main(model=model, loss_function=loss_function)
    
    
if __name__ == "__main__":
    args = argument_parsing()
    main(args)