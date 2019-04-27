# Load from run-main.sh settings and Greedy Decode

from pathlib import Path
import torch
import torch.nn as nn

from transformer.models import Transformer
from transformer.utils import get_pos

from main import argument_parsing
from dataloader import get_data

def get_model(sh_path):
    arguments = " ".join([s.strip() for s in Path(sh_path).read_text().replace("\\", "").replace('"', "").splitlines()[1:-1]])
    parser = argument_parsing(preparse=True)
    args = parser.parse_args(arguments.split())

    device = "cuda" if (torch.cuda.is_available() and args.use_cuda) else "cpu"
    (src, trg), (train, _, test), (train_loader, _, test_loader) = get_data(args)
    src_vocab_len = len(src.vocab.stoi)
    trg_vocab_len = len(trg.vocab.stoi)
    enc_max_seq_len = args.max_length
    dec_max_seq_len = args.max_length
    pad_idx = src.vocab.stoi.get("<pad>") if args.pad_idx is None else args.pad_idx
    pos_pad_idx = 0 if args.pos_pad_idx is None else args.pos_pad_idx

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
    if device == "cuda":
        model.load_state_dict(torch.load(args.save_path))
    else:
        model.load_state_dict(torch.load(args.save_path, map_location=torch.device(device)))
    
    return model, (src, trg), (test, test_loader)


def greedy_decode(model, max_length, src_data, trg_data, src_field, trg_field, 
                  enc_sos_idx=None, enc_eos_idx=None, dec_sos_idx=None, dec_eos_idx=None, device='cpu'):
    
    def get_real_attn(maxlen, attns_list):
        return [attn[:, :, :maxlen, :maxlen] for attn in attns_list]
        
    src_tensor = src_field.process([src_data]).to(device)
    trg_tensor = trg_field.process([trg_data]).to(device)
    # calculate real max lengths of sentences
    src_maxlen = src_tensor.ne(1).sum().item()
    trg_maxlen = trg_tensor.ne(1).sum().item()
    # get src sentence positions
    src_pos = get_pos(src_tensor, model.pad_idx, enc_sos_idx, enc_eos_idx)
    
    model.eval()
    decodes = []
    dec_tensor = torch.LongTensor([trg_field.vocab.stoi['<s>']]).unsqueeze(0).to(device)
    dec_pos = get_pos(dec_tensor, model.pad_idx, dec_sos_idx, dec_eos_idx)
    
    with torch.no_grad():
        enc_output, enc_self_attns = model.encoder(src_tensor, src_pos, return_attn=True)
        for i in range(max_length-1):
            dec_output, (dec_self_attns, dec_enc_attns) = \
                model.decoder(dec_tensor, dec_pos, src_tensor, enc_output, return_attn=True)
            output = model.projection(dec_output[:, -1])
            pred = output.argmax(-1)
            if pred.item() == dec_eos_idx:
                break
            else:
                dec_tensor = torch.cat([dec_tensor, pred.unsqueeze(-1)], dim=1)
                dec_pos = get_pos(dec_tensor, model.pad_idx, dec_sos_idx, dec_eos_idx)        
            
    attns_dict = {'enc_self_attns': get_real_attn(src_maxlen, enc_self_attns), 
                  'dec_self_attns': get_real_attn(trg_maxlen, dec_self_attns),
                  'dec_enc_attns': get_real_attn(trg_maxlen, dec_enc_attns)}
    return dec_tensor, attns_dict