import torch
from pathlib import Path
import numpy as np
from transformer.utils import get_pos
from tqdm import tqdm

class Trainer(object):
    """model trainer: torchtext"""
    def __init__(self, optimizer, train_loader, test_loader, n_step, device, save_path, 
                 enc_sos_idx=None, enc_eos_idx=None, dec_sos_idx=None, dec_eos_idx=None, metrics_method="acc", verbose=0):
        if not Path("trainlog/").exists():
            Path("trainlog").mkdir()
        self.save_path = save_path
        self.record_path = Path("trainlog/")/("train-log-"+ save_path.split("/")[-1].split('.')[0] +".txt")
        
        self.n_step = n_step
        self.device = device
        self.enc_sos_idx = enc_sos_idx
        self.enc_eos_idx = enc_eos_idx
        self.dec_sos_idx = dec_sos_idx
        self.dec_eos_idx = dec_eos_idx
        
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.metrics_method = metrics_method
        self.verbose = verbose
        self.tocpu = lambda x: x.cpu()

    def main(self, model, loss_function, rt_losses=False):
        import time
        
        start_time = time.time()
        
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        if self.metrics_method == "acc":
            lowest_metrics = 0.0
        elif self.metrics_method == "loss":
            lowest_metrics = 999
        else:
            assert False, "metrics_method = acc or loss"
        
        for step in range(1, self.n_step+1):
            train_loss, train_acc = self.train(model, loss_function, step)
            test_loss, test_acc = self.test(model, loss_function)
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            
            self._print(step, train_loss, test_loss, train_acc, test_acc)
            if self.metrics_method == "acc":
                test_metrics, test_metrics_list = test_acc, test_accs
            elif self.metrics_method == "loss":
                test_metrics, test_metrics_list = test_loss, test_losses
            lowest_metrics, early_break = self.save_model(model, test_metrics, test_metrics_list, lowest_metrics)
            if early_break:
                print(" - early break!!")
                break
                
        # Save Log for drawing graph
        np.savetxt("trainlog/losses-accs-" + self.save_path.split("/")[-1].split('.')[0] +".txt", 
                   np.array([train_losses, train_accs, test_losses, test_accs]), fmt="%.4e")
        # Time
        end_time = time.time()
        total_time = end_time-start_time
        hour = int(total_time // (60*60))
        minute = int((total_time - hour*60*60) // 60)
        second = total_time - hour*60*60 - minute*60
        txt = f"\nTraining Excution time with validation: {hour:d} h {minute:d} m {second:.4f} s"
        self._print_record(txt)
        
    def train(self, model, loss_function, step):
        """train model"""
        model.train()
        train_loss = 0
        n_correct = 0
        n_word = 0
        
        # setting iterator by verbose
        if self.verbose == 0:
            iterator = self.train_loader
        elif self.verbose == 1:
            iterator = tqdm(self.train_loader, desc=f"Training: {step}", total=len(self.train_loader))
        else:
            assert False, "set verbose 0 or 1"

        for batch in iterator:
            src, trg = batch.src, batch.trg
            batch_size = src.size(0)
            
            src_pos = get_pos(src, model.pad_idx, self.enc_sos_idx, self.enc_eos_idx)
            trg_pos = get_pos(trg, model.pad_idx, self.dec_sos_idx, self.dec_eos_idx)
            
            self.optimizer.zero_grad()
            output = model(src, src_pos, trg, trg_pos)
            
            real_trg = trg[:, 1:].contiguous()
            loss = loss_function(output, real_trg) 
            loss.backward()
            self.optimizer.step()
            # record
            train_loss += loss.item()
            pred = self.tocpu(output).view(-1, output.size(-1))
            n_correct += (pred.argmax(-1) == self.tocpu(real_trg).view(-1)).sum().item()
            n_word += trg.ne(model.pad_idx).sum().item()
            train_acc = n_correct / n_word
            
        return train_loss / n_word, train_acc
    
    def test(self, model, loss_function):
        """test model"""
        model.eval()
        test_loss = 0
        n_correct = 0
        n_word = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                src, trg = batch.src, batch.trg
                
                src_pos = get_pos(src, model.pad_idx, self.enc_sos_idx, self.enc_eos_idx)
                trg_pos = get_pos(trg, model.pad_idx, self.dec_sos_idx, self.dec_eos_idx)
                
                output = model(src, src_pos, trg, trg_pos)
                
                real_trg = trg[:, 1:].contiguous()
                loss = loss_function(output, real_trg) 
                
                # record
                test_loss += loss.item()
                pred = self.tocpu(output).view(-1, output.size(-1))
                n_correct += (pred.argmax(-1) == self.tocpu(real_trg).view(-1)).sum().item()
                n_word += trg.ne(model.pad_idx).sum().item()
                test_acc = n_correct / n_word
                
                
        return test_loss / n_word, test_acc
    
    def save_model(self, model, test_metrics, test_metrics_list, lowest_metrics):
        """early stopping"""
        early_break = False
        if len(test_metrics_list) >= 2:
            if self.metrics_method == "acc":
                if test_metrics >= lowest_metrics:
                    torch.save(model.state_dict(), self.save_path)
                    lowest_metrics = test_metrics
                    self._print_record("  - discard previous state, best model state saved!")
                if lowest_metrics == 1.0:
                    early_break = True
                    
            elif self.metrics_method == "loss":
                if test_metrics <= lowest_metrics:
                    torch.save(model.state_dict(), self.save_path)
                    lowest_metrics = test_metrics
                    self._print_record("  - discard previous state, best model state saved!") 
                if lowest_metrics == 0.0:
                    # TODO: change early break for 'loss' method
                    early_break = True
                    
        return lowest_metrics, early_break   
    
    def _print(self, step, train_loss, test_loss, train_acc, test_acc):
        """print log"""
        print_list = []
        print_list += [f"[{step}/{self.n_step}] Training Result"]
        print_list += [f"  - train_acc: {train_acc*100:.2f}%\t test_acc: {test_acc*100:.2f}%"]
        print_list += [f"  - train_ppl: {np.exp(min(train_loss, 100)):8.4f}\t test_ppl: {np.exp(min(test_loss, 100)):8.4f}"]
        for txt in print_list:
            self._print_record(txt)

    
    def _print_record(self, txt):
        print(txt)
        with open(self.record_path, "a") as config:
            config.write(txt+"\n")