import unittest
import model_evaluator as me
from torch import optim,nn
from torchvision import datasets, transforms
import torch


class TestModelEvaluator(unittest.TestCase):

  def setUp(self):
    self.loaded_dummy_dict = {'model_architecture':{'model':'a',
                               'optimizer':'b',
                               'criterion':'c'},
         'data':{'train_set':'d'},
        'training_parameters':{'no_of_steps_per_epoch':'e',
                               'device':'f',
                               'epochs':'g',
                               'time':'h'},
        'training_stats':{'total_train_time':'j',
                          'epoch_time':['k'],
                          'epoch_average_batch_loss':['l']}}
    self.model_evaluator_empty = me.Model('fashion_mnist','fashion_mnist_test.pkl')
    self.model = nn.Sequential(
                    nn.Linear(784,128),
                    nn.ReLU(),
                    nn.Linear(128,32),
                    nn.ReLU(),
                    nn.Linear(32,10),
                    nn.LogSoftmax(dim=1))
#defining the loss
    self.criterion = nn.NLLLoss()
    self.optimizer = optim.Adam(self.model.parameters(),lr=0.01)
    self.model_evaluator = self.model_evaluator_empty
    self.model_evaluator.reset_experiment()
    self.model_evaluator_empty.reset_experiment()
    self.epochs = 10
    self.cuda_is_available = True
    self.cuda_is_available = False
    
    self.transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    self.trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=self.transform)
    self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=True)


  
  def test_reset_experiment(self):
    self.model_evaluator.model = self.loaded_dummy_dict
    self.model_evaluator.add_model('test_model')
    self.model_evaluator.reset_experiment()
    self.assertEqual(self.model_evaluator.models,{})
    
  def test_remove_model_from_dict(self):
    self.model_evaluator.model = self.loaded_dummy_dict
    self.model_evaluator.add_model('fashion_mnist_2')
    self.model_evaluator.remove_model('fashion_mnist_2')
    self.assertEqual(self.model_evaluator.models,{})
    
  def test_validate_model_positive(self):
    self.model_evaluator.model = self.loaded_dummy_dict
    self.assertTrue(self.model_evaluator.validate_model())
    
  def test_validate_model_negative(self):
    self.model_evaluator.model = self.loaded_dummy_dict
    self.model_evaluator.model['model_architecture']['model'] = None
    self.assertFalse(self.model_evaluator.validate_model())
    
  def test_add_model(self):
    self.model_evaluator.model = self.loaded_dummy_dict
    self.model_evaluator.add_model('fashion_mnist_2')
    self.assertEqual(len(self.model_evaluator.models),1)
    
  def test_add_model_architecture_model_notnone(self):
    self.model_evaluator.add_model_architecture(self.model,self.criterion,self.optimizer)
    self.assertIsNotNone(self.model_evaluator.model['model_architecture']['model'])
    
  def test_add_model_architecture_criterion_notnone(self):
    self.model_evaluator.add_model_architecture(self.model,self.criterion,self.optimizer)
    self.assertIsNotNone(self.model_evaluator.model['model_architecture']['criterion'])
    
  def test_add_model_architecture_optimizer_notnone(self):
    self.model_evaluator.add_model_architecture(self.model,self.criterion,self.optimizer)
    self.assertIsNotNone(self.model_evaluator.model['model_architecture']['optimizer'])
    
  def test_add_training_parameters_epochs_notnone(self):
    self.model_evaluator.add_training_parameters(self.epochs,self.trainloader,self.model)
    self.assertIsNotNone(self.model_evaluator.model['training_parameters']['epochs'])
    
  def test_add_training_parameters_no_of_batches_notnone(self):
    self.model_evaluator.add_training_parameters(self.epochs,self.trainloader,self.model)
    self.assertIsNotNone(self.model_evaluator.model['training_parameters']['no_of_steps_per_epoch'])
    
  def test_add_training_parameters_device_notnone(self):
    self.model_evaluator.add_training_parameters(self.epochs,self.trainloader,self.model)
    self.assertIsNotNone(self.model_evaluator.model['training_parameters']['device'])
    
  def test_add_training_parameters_time_notnone(self):
    self.model_evaluator.add_training_parameters(self.epochs,self.trainloader,self.model)
    self.assertIsNotNone(self.model_evaluator.model['training_parameters']['time'])
    
  def test_add_training_parameters_device_cpu(self):
    self.model_evaluator.add_training_parameters(self.epochs,self.trainloader,self.model)
    self.assertEqual(self.model_evaluator.model['training_parameters']['device'],'cpu')
  
  @unittest.skip("GPU Unavailable to test the scenario")  
  def test_add_training_parameters_device_gpu(self):
    self.model = self.model.cuda()
    self.model_evaluator.add_training_parameters(self.epochs,self.trainloader,self.model)
    self.assertEqual(self.model_evaluator.model['training_parameters']['device'],'gpu')
    
if __name__ == '__main__':
	unittest.main()
