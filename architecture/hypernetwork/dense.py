import numpy
import torch
from torch import nn


class DenseHypernetwork(nn.Module):

	def __init__(self, base_model, input_shape):
		super().__init__()
		self.base_model = base_model
		self.named_submodels = nn.ModuleDict()
		in_features = numpy.prod(input_shape)

		for name, parameter in self.base_model.named_parameters():
			self.named_submodels[name] = nn.Linear(
				in_features=in_features, 
				out_features=numpy.prod(parameter.size()), 
				bias=True)


	# TODO: Generalize forward method to accomodate base_models 
	# that act on multiple tensors. 
	# TODO: Support batch sizes > 1. Either average over subodel 
	# outputs per batch or compute each batch example individually.
	def forward(self, x):
		x = torch.flatten(x, start_dim=1, end_dim=-1)

		for name in self.named_submodels:
			# Reshape submodel output
			submodel_out = self.named_submodels[name](x)
			submodel_out = torch.reshape(submodel_out, getattr(self.base_model, name).size())
			
			# Set base model weights to submodel output.
			self.base_model.__dict__[name] = submodel_out

		return self.base_model(x)