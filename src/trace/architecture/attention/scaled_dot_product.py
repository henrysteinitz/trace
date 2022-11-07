import numpy
import torch
from torch import nn
from torchtyping import TensorType
from typeguard import typechecked


class ScaledDotProductAttention(nn.Module):

	@typechecked
	def forward(self, 
			query: TensorType["batch", "d_k"], 
			keys: TensorType["batch", "d", "d_k"], 
			values: TensorType["batch", "d", "d_v"]):
		d_k = query.size()[-1]
		x = torch.matmul(query, torch.transpose(keys, dim0=-1, dim1=-2))
		x = (1 / numpy.sqrt(d_k)) * x
		x = torch.softmax(x, dim=2)
		return torch.matmul(x, values)