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
	def forward(self, x):
		x = torch.flatten(x, start_dim=1, end_dim=-1)

		base_model_state = self.base_model.state_dict()
		for name in self.named_submodels:
			# Reshape submodel output
			submodel_out = self.named_submodels[name](x)
			submodel_out = torch.reshape(submodel_out, base_model_state[name].size())
			
			# Set base model weights to submodel output.
			base_model_state[name] = submodel_out
		
		self.base_model.load_state_dict(base_model_state)
		return self.base_model(x)