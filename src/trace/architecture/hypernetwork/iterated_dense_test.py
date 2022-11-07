import torch
import unittest
from torch import nn, optim
from torch.testing import assert_close
from torch.utils.data import Dataset, DataLoader

from dense_test import SyntheticLinearRegressionDataset
from iterated_dense import IteratedDenseHypernetwork


class TestIteratedDenseHypernetwork(unittest.TestCase):

	def test_forward_shape(self):
		base_model = nn.Linear(in_features=4, out_features=3, bias=True)
		model = IteratedDenseHypernetwork(base_model, num_iterations=4, input_shape=[4])
		x = torch.Tensor([[1.0, 2.0, 3.0, 4.0]])
		self.assertEqual(model(x).size(), torch.Size([1, 3]))


	def test_linear_regression(self):
		base_model = nn.Linear(in_features=2, out_features=2, bias=True)
		model = IteratedDenseHypernetwork(base_model, num_iterations=2, input_shape=[2])
		
		A = torch.Tensor([
			[1.0, 2.0],
			[3.0, 4.0]
		])
		b = torch.Tensor([7.0, 8.0])

		# TODO: Move training loop into helper function.
		dataset = SyntheticLinearRegressionDataset(A=A, b=b, size=10000)
		data = DataLoader(dataset, batch_size=1, shuffle=True)
		optimizer = optim.SGD(model.parameters(), lr=.07, momentum=0.9)
		loss_fn = nn.HuberLoss()
		for _ in range(7):
			for X, Y in data:
				optimizer.zero_grad()
				Y_prime = model(X)
				loss = loss_fn(Y_prime, Y)
				loss.backward()
				optimizer.step()

		# The iterated hypernetwork should learn the function y = Ax + b, so [1, 1] should
		# produce [10, 15].
		assert_close(model(torch.Tensor([[1., 1.]])), torch.Tensor([[10., 15.]]), atol=.001, rtol=0)


if __name__ == '__main__':
    unittest.main()