import os
import torch

class ModelEvaluator:
  

  def __init__(self,dataset,file_path):
    self.dataset = dataset
    self.file_path = file_path
    try:
      self.models = torch.load(file_path)
    except FileNotFoundError:
      self.models = {}
      
  def init_model(self):
    model = {'model_architecture':{'model':None,
                               'optimizer':None,
                               'criterion':None},
         'data':{'train_set':None},
        'training_parameters':{'no_of_steps_per_epoch':None,
                               'device':None,
                               'epochs':None,
                               'time':None},
        'training_stats':{'total_train_time':None,
                          'epoch_time':[],
                          'epoch_average_batch_loss':[]}}
    return model

  def load_model(self,file_path):
    try:
      self.models = torch.load(file_path)
    except FileNotFoundError:
      self.models = {}
    
  def add_model(self,model_name,model_dict):
    self.models[model_name] = model_dict
    torch.save(self.models,self.file_path)
      
  def remove_model(self,model_name):
    del self.models[model_name]
  
  def is_empty(self):
    return len(self.models) == 0

  def list_models(self):
    if self.is_empty():
      print("No Models added")
    for model_name in self.models.keys():
      print(model_name)

  def reset_experiment(self):
    """
    Empties the experiment file.
    1. Delete the file
    """
    if os.path.exists(self.file_path):
      os.remove(self.file_path)
      self.models = {}

