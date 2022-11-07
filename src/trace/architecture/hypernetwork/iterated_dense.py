from dense import DenseHypernetwork


class IteratedDenseHypernetwork(nn.Module):

	def __init__(self, base_model, num_iterations, input_size):
		self.models = ModuleList([base_model])
		for i in range(num_iterations):
			self.model.append(DenseHypernetwork(self.models[-1]))


	def forward(self, x):
		return self.models[-1](x)
