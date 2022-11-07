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
			# Module names cannot contain '.'.
			modified_name = name.replace(".", "_")
			self.named_submodels[modified_name] = nn.Linear(
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
			# Access nested module.
			module_path = name.split("_")
			base_model = self.base_model
			for i in range(len(module_path) - 1):
				base_model = base_model._modules[module_path[i]]

			# Reshape submodel output.
			submodel_out = self.named_submodels[name](x)
			submodel_out = torch.reshape(submodel_out, getattr(base_model, module_path[-1]).size())

			# Set base model weights to submodel output. We need to attach the tensor directly
			# to base_model.__dict__ because Module.__setattr__ enforces that existing parameters 
			# remain parameters. We cannot contruct a parameter from our tensor because that will
			# erase its computational history and break future gradient computations. 
			base_model.__dict__[module_path[-1]] = submodel_out

		return self.base_model(x)