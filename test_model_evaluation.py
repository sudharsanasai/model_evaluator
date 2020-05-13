import unittest
import model_evaluator as me

class TestModelEvaluator(unittest.TestCase):
  
  
  def test_reset_experiment(self):
    model_evaluator = me.ModelEvaluator('fashion_mnist','fashion_mnist_test.pkl')
    model_evaluator.reset_experiment()
    empty_dict = {}
    self.assertEqual(model_evaluator.models,empty_dict)
    
  def test_remove_model_from_dict(self):
    model_evaluator = me.ModelEvaluator('fashion_mnist','fashion_mnist_test.pkl')
    model_evaluator.reset_experiment()
    model_evaluator.add_model('fashion_mnist_2','test_model')
    model_evaluator.remove_model('fashion_mnist_2')
    self.assertEqual(model_evaluator.models,{})
    
if __name__ == '__main__':
	unittest.main()
