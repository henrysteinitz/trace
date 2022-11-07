import numpy
import torch
from torch import nn
from torchtyping import TensorType
from typeguard import typechecked


class ContrastiveAttention(nn.Module):

	@typechecked
	def forward(self, 
			query: TensorType["batch", "d_m"], 
			memory: TensorType["batch", "d", "d_m"]):
		x = torch.matmul(query, torch.transpose(memory, dim0=-1, dim1=-2))
		x = nn.functional.softmin(x, dim=2)
		return torch.matmul(x, memory)