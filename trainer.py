import torch
from transformer.utils import get_pos
from tqdm import tqdm

class Trainer(object):
    """model trainer: torchtext"""
    def __init__(self, optimizer, train_loader, test_loader, n_step, device, save_path, n_check=3, patient=5):
        import os
        if not os.path.isdir(os.path.split(save_path)[0]):
            os.mkdir(os.path.split(save_path)[0])
            
        self.save_path = save_path
        self.n_step = n_step
        self.device = device
        
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        # early stopping
        self.n_check = n_check
        self.patient = patient
        self.check_increase = lambda L: all(x <= y for x, y in zip(L, L[1:]))

    def main(self, model, loss_function, early_stop=True, rt_losses=False):
        import time
        
        start_time = time.time()
        
        train_losses = []
        test_losses = []
        lowest_loss = 999
        wait = 0
        
        for step in range(1, self.n_step+1):
            train_loss = self.train(model, loss_function, step)
            test_loss = self.test(model, loss_function)
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            self._print(step, train_loss, test_loss)
            
            if len(test_losses) >= self.patient:
                
                if test_loss <= lowest_loss:
                    lowest_loss = test_loss
                    torch.save(model.state_dict(), self.save_path)
                    print("discard previous state, best model state saved!")
                        
                if early_stop and (wait < self.patient):
                    if self.check_increase(test_losses[:-self.n_check]):
                        wait += 1
                elif early_stop and (wait >= self.patient):
                    print("*** Early Stopped! ***")
                    break
                    
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
        for batch in tqdm(self.train_loader, desc=f"Training: {step}", total=len(self.train_loader)):
            src_pos = get_pos(batch.src)
            trg_pos = get_pos(batch.trg)
            
            self.optimizer.zero_grad()
            output = model(batch.src, src_pos, batch.trg, trg_pos)
            loss = loss_function(output, batch.trg)
            loss.backward()
            self.optimizer.step()
            # record
            train_loss += loss.item()
            
        return train_loss / len(self.train_loader.dataset)
    
    def test(self, model, loss_function):
        """test model"""
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                src_pos = get_pos(batch.src)
                trg_pos = get_pos(batch.trg)
                
                output = model(batch.src, src_pos, batch.trg, trg_pos)
                loss = loss_function(output, batch.trg)
                test_loss += loss.item()
                
        return test_loss / len(self.test_loader.dataset)
    
    def _print(self, step, train_loss, test_loss):
        """print log"""
        print(f"[{step}/{self.n_step}] train_loss: {train_loss:.4f}\t test_loss: {test_loss:.4f}")