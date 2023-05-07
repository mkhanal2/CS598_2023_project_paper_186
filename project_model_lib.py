from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import torch

MODEL_SAVE_PATH = "./drive/MyDrive/DLHC-Data/pre_processed_data/models/"

class CustomDataset(Dataset):
    def __init__(self, x, x_entity, y):
        self.x = x
        self.x_entity = x_entity
        self.y = y
    
    def __len__(self):
        output = len(self.y)
        return output
    
    def __getitem__(self, index):
        return (self.x[index], self.x_entity[index] , self.y[index])


class BaseModel (nn.Module):
    def __init__(self, input_size, hidden_size):
      super().__init__()
      #super(BaseModel, self).__init__()
      self.hidden_size = hidden_size
      self.gru = nn.GRU(input_size, hidden_size, num_layers = 1, batch_first=True)
      self.fc1 = nn.Linear(in_features=hidden_size, out_features=256)
      self.relu1 = nn.ReLU()
      self.do = nn.Dropout(p=0.2)
      self.fc2 = nn.Linear(in_features=256, out_features=1, bias=False)
      self.sigmoid = nn.Sigmoid()

      # initialize the weights for the GRU layer
      for name, param in self.gru.named_parameters():
        if 'weight_ih' in name:
          nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
          nn.init.orthogonal_(param)

      # initialize the weights for the linear layer
      nn.init.xavier_uniform_(self.fc1.weight)
      nn.init.zeros_(self.fc1.bias)
      nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x, x_embd):
      x, _ = self.gru(x)
      x = x[:,23,:]
      x = self.fc1(x)
      x = self.relu1(x)
      x = self.do(x)
      x = self.fc2(x)
      x = self.sigmoid(x)
      return x

class MultiModel (nn.Module):
    def __init__(self, input_size, hidden_size, filter_size):
        super().__init__()
        #super(MultiModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers = 1, batch_first=True)
        self.cnn1 = nn.Conv1d(in_channels = 100, out_channels = filter_size, kernel_size = 3, stride = 1, padding = 0)
        self.cnn2 = nn.Conv1d(in_channels = filter_size, out_channels = filter_size*2, kernel_size = 3, stride = 1, padding = 0)
        self.cnn3 = nn.Conv1d(in_channels = filter_size*2, out_channels = filter_size*3, kernel_size = 3, stride = 1, padding = 0)
        self.maxpoll = nn.AdaptiveMaxPool1d(1)

        self.max1 = nn.MaxPool1d(kernel_size = 3, stride=1)
        self.max2 = nn.MaxPool1d(kernel_size = 3, stride=1)
        self.max3 = nn.MaxPool1d(kernel_size = 3, stride=1)

        self.fc1 = nn.Linear(in_features=(hidden_size + 96), out_features=512)
        self.relu1 = nn.ReLU()
        self.do = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=512, out_features=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        # initialize the weights for the GRU layer
        for name, param in self.gru.named_parameters():
          if 'weight_ih' in name:
            nn.init.xavier_uniform_(param)
          elif 'weight_hh' in name:
            nn.init.orthogonal_(param)

        # initialize the weights for the linear layer
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Initialize weights
        #init.kaiming_uniform_(self.cnn1.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_uniform_(self.cnn1.weight)
        init.constant_(self.cnn1.bias, 0.0)
        #init.kaiming_uniform_(self.cnn2.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_uniform_(self.cnn2.weight)
        init.constant_(self.cnn2.bias, 0.0)
        #init.kaiming_uniform_(self.cnn3.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_uniform_(self.cnn3.weight)
        init.constant_(self.cnn3.bias, 0.0)
        
    def forward(self, x, x_embd):
      #s = x.size()[0]
      #h0 = torch.randn((1,s,self.hidden_size),requires_grad=True)
      x, _ = self.gru(x)
      x = x[:,23,:]

      x_embd = torch.transpose(x_embd, 1, 2)
      x_embd = F.relu(self.cnn1(x_embd))
      x_embd = F.relu(self.cnn2(x_embd))
      x_embd = F.relu(self.cnn3(x_embd))
      
      #print(x_embd.size())
      x_embd = torch.max(x_embd, 2).values
      #x_embd = torch.mean(x_embd, 2)
      
      x = torch.cat((x, x_embd), 1)

      x = self.fc1(x)
      x = self.relu1(x)
      x = self.do(x)
      x = self.fc2(x)
      x = self.sigmoid(x)
      return x

class TransformerModel (nn.Module):
    def __init__(self, input_size, num_heads, num_layers):
      super().__init__()
      #super(MultiModel, self).__init__()
      
      self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(input_size, num_heads),num_layers)
      
      self.fc1 = nn.Linear(in_features=input_size, out_features=256)
      self.do = nn.Dropout(p=0.2)
      self.fc2 = nn.Linear(in_features=256, out_features=1, bias=False)
      self.sigmoid = nn.Sigmoid()
      
      # initialize the weights for the linear layer
      nn.init.xavier_uniform_(self.fc1.weight)
      nn.init.zeros_(self.fc1.bias)
      nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x, x_embd):
      x = self.transformer_encoder(x) #torch.Size([128, 100, 64])
      x = x[:,23,:]

      x = F.relu(self.fc1(x))
      x = self.do(x)
      x = self.fc2(x)
      x = self.sigmoid(x)
      return x

def eval_model(model, val_loader, loss_function):    
    model.eval()
    y_pred = torch.LongTensor()
    y_true = torch.LongTensor()
    y_score = torch.Tensor()
    validation_loss = 0

    for x, x_embd, y in val_loader:
        y_hat = model(x,x_embd)
        #all_y_true = y
        #y_hat = y_hat.reshape(y.shape)
        all_y_true = y.reshape(y_hat.shape)
        loss = loss_function(y_hat,all_y_true)
        validation_loss += loss.item()

        y_score = torch.cat((y_score,  y_hat.detach().to('cpu')), dim=0)
        #y_hat = (y_hat > 0.5).int()
        y_hat = torch.Tensor([1 if i>=0.5 else 0 for i in y_hat])
        y_pred = torch.cat((y_pred,  y_hat.detach().to('cpu')), dim=0)
        y_true = torch.cat((y_true, all_y_true.detach().to('cpu')), dim=0)
        
    y_score = torch.squeeze(y_score).tolist()
    y_pred = torch.squeeze(y_pred).tolist()
    y_true = torch.squeeze(y_true).tolist()

    validation_loss = validation_loss / len(val_loader)
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    
    return auroc, auprc, f1, acc, validation_loss

class EarlyStopper:
  def __init__(self, patience=1, min_delta=0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation_loss = np.inf

  def early_stop(self, validation_loss):
    if validation_loss < self.min_validation_loss:
      self.min_validation_loss = validation_loss
      self.counter = 0
    elif validation_loss > (self.min_validation_loss + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False

class BestModelCheckPoint ():
  def __init__(self, save_path):
    self.path = save_path

  def save_best_model(self, model):
    torch.save({'model_state_dict': model.state_dict()}, self.path)
  
  def get_best_model(self, model):
    checkpoint = torch.load(self.path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

class SaveAllResults():
  def __init__(self, save_path):
    self.path = save_path
    self.all_results = pd.DataFrame()

  def save_result(self, var_name, model_name, run_id, auroc, auprc, f1, acc): 
    data = {  'var_name'  : [var_name]
            , 'model_name': [model_name]
            , 'run_id'    : [run_id]
            , 'auroc'     : [auroc]
            , 'auprc'     : [auprc]
            , 'f1'        : [f1]
            , 'acc'       : [acc]
            }
    df = pd.DataFrame.from_dict(data)
    if(len(self.all_results) > 0):
      self.all_results = pd.concat([self.all_results, df], axis=0, ignore_index=True)
    else:
      self.all_results = df
    pd.to_pickle(self.all_results, self.path)
    
  def get_result(self):
    if (len(self.all_results) > 0):
      return self.all_results
    else:
      self.all_results = pd.read_pickle(self.path)
    return self.all_results

def train_model(model, train_loader, valid_loader, model_name, n_epochs = 100, lr = 0.001, patience = 3, lr_scheduler=False): 
  criterion_trn = nn.BCELoss()
  optimizer_trn = torch.optim.Adam(model.parameters(), lr=lr, weight_decay= 0.01)
  training_loss = []
  validation_loss = []
  early_stopper = EarlyStopper(patience=patience, min_delta=0.00)
  if (lr_scheduler):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_trn, mode='min', factor=0.2, patience=1, threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0.00001, eps=1e-04, verbose=False)
  full_path_checkpoint = MODEL_SAVE_PATH + model_name
  best_model_checkpoint = BestModelCheckPoint(full_path_checkpoint)
  tst_rslt=[]
  for epoch in range(n_epochs):
    step_loss = []
    model.train()
    train_loss = 0
    for x,x_embd, y in train_loader:
      optimizer_trn.zero_grad()
      y_hat = model.forward(x,x_embd)
      #print(y_hat.size())
      y_hat = y_hat.reshape(y.shape)
      all_y_true = y
      #all_y_true = y.reshape(y_hat.shape)
      loss = criterion_trn(y_hat,all_y_true)
      #print(loss)
      loss.backward()
      optimizer_trn.step()
      train_loss += loss.item()
    train_loss = train_loss / len(train_loader)
    training_loss.append(train_loss)
    #print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))
    auroc, auprc, f1, acc, val_loss = eval_model(model, valid_loader, criterion_trn)
    validation_loss.append(val_loss)
    improved = 0.0
    lr = optimizer_trn.param_groups[0]['lr']
    if (val_loss <= min(validation_loss)):
      best_model_checkpoint.save_best_model(model)
      improved = 1.0
    print('Epoch: {} \t Training Loss: {:.6f}, lr:{}, Validation (auroc: {:.2f}, auprc:{:.2f}, f1: {:.2f}, acc: {:.2f}, val_loss:{:.2f}, improved:{})'.format(epoch+1, train_loss, lr, auroc, auprc, f1, acc, val_loss, improved))
    if (lr_scheduler):
      scheduler.step(val_loss)
    if early_stopper.early_stop(val_loss):
      break
  model = best_model_checkpoint.get_best_model(model)
  return model
