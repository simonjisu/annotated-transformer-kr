import torch
import numpy as np
from transformer.utils import get_pos
from tqdm import tqdm

class Trainer(object):
    """model trainer: torchtext"""
    def __init__(self, optimizer, train_loader, test_loader, n_step, device, save_path, verbose=0):
        
        self.save_path = save_path
        self.n_step = n_step
        self.device = device
        
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.verbose = verbose

    def main(self, model, loss_function, rt_losses=False):
        import time
        
        start_time = time.time()
        
        train_losses = []
        test_losses = []
        lowest_metrics = 999
        
        for step in range(1, self.n_step+1):
            train_loss = self.train(model, loss_function, step)
            test_loss = self.test(model, loss_function)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            self._print(step, train_loss, test_loss)
            lowest_metrics = self.save_model(model, test_loss, test_losses, lowest_metrics)

        end_time = time.time()
        total_time = end_time-start_time
        hour = int(total_time // (60*60))
        minute = int((total_time - hour*60*60) // 60)
        second = total_time - hour*60*60 - minute*60
        print('\nTraining Excution time with validation: {:d} h {:d} m {:.4f} s'.format(hour, minute, second))
        
    def train(self, model, loss_function, step):
        """train model"""
        model.train()
        train_loss = 0
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
            
            src_pos = get_pos(src)
            trg_pos = get_pos(trg)
            
            self.optimizer.zero_grad()
            output = model(src, src_pos, trg, trg_pos)
            loss = loss_function(output, trg) 
            loss.backward()
            self.optimizer.step()
            # record
            train_loss += loss.item()
            n_word += trg.ne(model.pad_idx).sum().item()
            
        return train_loss / n_word
    
    def test(self, model, loss_function):
        """test model"""
        model.eval()
        test_loss = 0
        n_word = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                src, trg = batch.src, batch.trg
                
                src_pos = get_pos(src)
                trg_pos = get_pos(trg)
                
                output = model(src, src_pos, trg, trg_pos)
                loss = loss_function(output, trg) 
                test_loss += loss.item()
                n_word += trg.ne(model.pad_idx).sum().item()
                
        return test_loss / n_word
    
    def _print(self, step, train_loss, test_loss):
        """print log"""
        print("[{}/{}] train_ppl: {:8.5f}\t test_ppl: {:8.5f}".format(
            step, self.n_step, np.exp(min(train_loss, 100)), np.exp(min(test_loss, 100))))
    
    def save_model(self, model, test_metrics, test_metrics_list, lowest_metrics):
        """early stopping"""
        if len(test_metrics_list) >= 2:
            if test_metrics <= lowest_metrics:
                torch.save(model.state_dict(), self.save_path)
                lowest_metrics = test_metrics
                print("discard previous state, best model state saved!")

        return lowest_metrics