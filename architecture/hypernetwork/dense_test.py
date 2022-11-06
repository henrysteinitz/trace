import torch
import unittest
from torch import nn
from torch.testing import assert_close

from dense import DenseHypernetwork


class TestDenseHypernetwork(unittest.TestCase):

	def test_forward_shape(self):
		base_model = nn.Linear(in_features=4, out_features=3, bias=True)
		model = DenseHypernetwork(base_model, input_shape=[4])
		x = torch.Tensor([[1.0, 2.0, 3.0, 4.0]])
		self.assertEqual(model(x).size(), torch.Size([1, 3]))


	def test_forward_value(self):
		base_model = nn.Linear(in_features=1, out_features=1, bias=True)
		model = DenseHypernetwork(base_model, input_shape=[1])
		
		state = model.state_dict()
		state["named_submodels.weight.weight"] = torch.Tensor([[1.0]])
		state["named_submodels.weight.bias"] = torch.Tensor([2.0])
		state["named_submodels.bias.weight"] = torch.Tensor([[2.0]])
		state["named_submodels.bias.bias"] = torch.Tensor([2.0])
		model.load_state_dict(state)

		# When X = 1, the hypernetwork's weight submodel will ouput 1(1) + 2 = 3.
		# Similarly the bias submodel will output 2(1) + 2 = 4. The output of 
		# the base model should then be 3(1) + 4 = 7. 
		x = torch.Tensor([[1.0]])
		self.assertEqual(model(x), torch.Tensor([[7.0]]))


if __name__ == '__main__':
    unittest.main()