import unittest
import model_evaluator as me

class TestModelEvaluator(unittest.TestCase):
  def __init__(self, *args, **kwargs):
        super(TestModelEvaluator, self).__init__(*args, **kwargs)
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
        
  
  def test_reset_experiment(self):
    model_evaluator = self.model_evaluator_empty
    model_evaluator.model = self.loaded_dummy_dict
    model_evaluator.add_model('test_model')
    model_evaluator.reset_experiment()
    self.assertEqual(model_evaluator.models,{})
    
  def test_remove_model_from_dict(self):
    model_evaluator = self.model_evaluator_empty
    model_evaluator.reset_experiment()
    model_evaluator.model = self.loaded_dummy_dict
    model_evaluator.add_model('fashion_mnist_2')
    model_evaluator.remove_model('fashion_mnist_2')
    self.assertEqual(model_evaluator.models,{})
    
  def test_validate_model_positive(self):
    model_evaluator = self.model_evaluator_empty
    model_evaluator.reset_experiment()
    model_evaluator.model = self.loaded_dummy_dict
    self.assertTrue(model_evaluator.validate_model())
    
    
  def test_validate_model_negative(self):
    model_evaluator = self.model_evaluator_empty
    model_evaluator.reset_experiment()
    model_evaluator.model = self.loaded_dummy_dict
    model_evaluator.model['model_architecture']['model'] = None
    self.assertFalse(model_evaluator.validate_model())
    
  def test_add_model(self):
    model_evaluator = self.model_evaluator_empty
    model_evaluator.reset_experiment()
    model_evaluator.model = self.loaded_dummy_dict
    model_evaluator.add_model('fashion_mnist_2')
    self.assertEqual(len(model_evaluator.models),1)
    
    
if __name__ == '__main__':
	unittest.main()
