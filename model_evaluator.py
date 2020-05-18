import os
import torch

class Model:
  

  def __init__(self,dataset,file_path):
    self.dataset = dataset
    self.file_path = file_path
    try:
      self.models = torch.load(file_path)
    except FileNotFoundError:
      self.models = {}
    self.model = {'model_architecture':{'model':None,
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

  def load_model(self,file_path):
    try:
      self.models = torch.load(file_path)
    except FileNotFoundError:
      self.models = {}
    
  def add_model(self,model_name):
    if self.validate_model():
      self.models[model_name] = self.model
      torch.save(self.models,self.file_path)
    else:
      print("Model not added.")
      
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

  def validate_model(self):
    is_valid = True
    
    for model_architecture_param in self.model['model_architecture'].keys():
      if self.model['model_architecture'][model_architecture_param] is None:
        print(model_architecture_param+" is not set.")
        is_valid = False
    
    for training_param in self.model['training_parameters'].keys():
      if self.model['training_parameters'][training_param] is None:
        print(training_param+" is not set.")
        is_valid = False
        
    if self.model['data']['train_set'] is None:
      print('Training Set is not set.')
      is_valid = False
        
    if len(self.model['training_stats']['epoch_time']) == 0:
      print('Epoch Timings is not set.')
      is_valid = False
        
    if len(self.model['training_stats']['epoch_average_batch_loss']) == 0:
      print('Epoch losses are not set.')
      is_valid = False
        
    return is_valid
  
  def add_model_architecture(self,model,criterion,optimizer):
    self.model['model_architecture']['model'] = str(model)
    self.model['model_architecture']['criterion'] = str(criterion)
    self.model['model_architecture']['optimizer'] = str(optimizer)