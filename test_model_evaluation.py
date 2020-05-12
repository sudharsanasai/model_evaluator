import unittest
import model_evaluator as me

class TestModelEvaluator(unittest.TestCase):
  def test_reset_experiment(self):
    model_evaluator = me.ModelEvaluator('fashion_mnist',"/content/drive/My Drive/Masters/Deep Learning/model_evaluator/model_evaluator/fashion_mnist_test.pkl")
    model_evaluator.reset_experiment()
    empty_dict = {}
    self.assertEqual(model_evaluator.models,empty_dict)

if __name__ == '__main__':
	unittest.main()