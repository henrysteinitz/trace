import torch
import unittest
from torch import nn

from iterated_dense import IteratedDenseHypernetwork


class TestIteratedDenseHypernetwork(unittest.TestCase):

	def test_forward_shape(self):
		base_model = nn.Linear(in_features=4, out_features=3, bias=True)
		model = IteratedDenseHypernetwork(base_model, num_iterations=4, input_shape=[4])
		x = torch.Tensor([[1.0, 2.0, 3.0, 4.0]])
		self.assertEqual(model(x).size(), torch.Size([1, 3]))


if __name__ == '__main__':
    unittest.main()