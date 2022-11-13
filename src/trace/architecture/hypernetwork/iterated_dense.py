from torch import nn
from trace.architecture.hypernetwork.dense import DenseHypernetwork


class IteratedDenseHypernetwork(nn.Module):

	def __init__(self, base_model, num_iterations, input_shape):
		super().__init__()
		self.models = nn.ModuleList([base_model])
		for i in range(num_iterations):
			self.models.append(DenseHypernetwork(self.models[-1], input_shape=input_shape))


	def forward(self, x):
		return self.models[-1](x)