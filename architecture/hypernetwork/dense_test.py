import torch
import unittest
from torch import nn, optim
from torch.testing import assert_close
from torch.utils.data import Dataset, DataLoader

from dense import DenseHypernetwork

# TODO: Move into evaluation/dataset/regression
class SyntheticLinearRegressionDataset(Dataset):

    def __init__(self, A, b, size):
        self.X = [torch.rand_like(b) for _ in range(size)]
        self.Y = [torch.matmul(A, x) + b for x in self.X]


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SyntheticQuadraticRegressionDataset(Dataset):

    def __init__(self, A, b, size):
        self.X = [torch.rand_like(b) for _ in range(size)]
        self.Y = [torch.matmul(torch.matmul(A, x), x) + torch.matmul(A, x) + b for x in self.X]


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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


	def test_linear_regression(self):
		# A hypernetwork with dense submodels should be strictly more powerful 
		# than the base model. The reason is that learning the base model weights 
		# is equivalent to learning the submodel biases. In fact, this principle 
		# should generalize to any submodel with dense output layers.
		base_model = nn.Linear(in_features=2, out_features=2, bias=True)
		model = DenseHypernetwork(base_model, input_shape=[2])
		
		A = torch.Tensor([
			[1.0, 2.0],
			[3.0, 4.0]
		])
		b = torch.Tensor([7.0, 8.0])

		# TODO: Move training loop into helper function.
		dataset = SyntheticLinearRegressionDataset(A=A, b=b, size=10000)
		data = DataLoader(dataset, batch_size=1, shuffle=True)
		optimizer = optim.SGD(model.parameters(), lr=.07, momentum=0.9)
		loss_fn = nn.MSELoss()
		for _ in range(3):
			for X, Y in data:
				optimizer.zero_grad()
				Y_prime = model(X)
				loss = loss_fn(Y_prime, Y)
				loss.backward()
				optimizer.step()

		# The hypernetwork should learn the function y = Ax + b, so [1, 1] should
		# produce [10, 15].
		assert_close(model(torch.Tensor([[1., 1.]])), torch.Tensor([[10., 15.]]), atol=.001, rtol=0)


	def test_quadradic_regression(self):
		# A hypernetwork does not need nonlinearities to compute nonlinear functions. In fact,
		# a deep linear hypernetwork with n layers can compute n-degree polynomials.
		base_model = nn.Sequential(
			nn.Linear(in_features=2, out_features=2, bias=True),
			nn.Linear(in_features=2, out_features=2, bias=True)
		)
		model = DenseHypernetwork(base_model, input_shape=[2])

		A = torch.Tensor([
			[1.0, 2.0],
			[-1.0, -1.0]
		])
		b = torch.Tensor([3.0, 4.0])

		# TODO: Move training loop into helper function.
		dataset = SyntheticQuadraticRegressionDataset(A=A, b=b, size=10000)
		data = DataLoader(dataset, batch_size=1, shuffle=True)
		optimizer = optim.SGD(model.parameters(), lr=.03, momentum=0.9)
		loss_fn = nn.HuberLoss()
		for _ in range(7):
			for X, Y in data:
				optimizer.zero_grad()
				Y_prime = model(X)
				loss = loss_fn(Y_prime, Y)
				loss.backward()
				optimizer.step()

		# (Ax)x + Ax + b for x = [1, 1] yields [7, 3]. The tolerance is increased to 
		# reduce the total number of epochs and speed up the test.
		assert_close(model(torch.Tensor([[1., 1.]])), torch.Tensor([[7., 3.]]), atol=.01, rtol=0.)


if __name__ == '__main__':
    unittest.main()