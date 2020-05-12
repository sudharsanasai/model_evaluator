
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

  def load_model(self,file_path):
    try:
      self.models = torch.load(file_path)
    except FileNotFoundError:
      self.models = {}
    
  def add_model(self,model_name,model_dict):
      self.models[model_name] = model_dict
      torch.save(self.models,self.file_path)

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

  def retreive_loss_over(self,model_name):
    if model_name in self.model.keys():
      return self.models[model_name]['training_stats']['epoch_average_batch_loss']
    else:
      print('Could not find '+ model_name)
