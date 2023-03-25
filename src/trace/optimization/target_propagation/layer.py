import torch
from torch import nn, autograd


# Implementation assumes parameters are attached directly to the model.
class TargetPropagationLayer(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.custom_forward = None
		self.child_forward = self.forward
		self.forward = self.forward_override
		self.forward_inputs = None
		self.ordered_parameters = None
		

	def _attach_parameters(self):
		for name, parameter in self.named_parameters():
			self.register_parameter(name=name, param=parameter)
			parameter.parameter_name = name

		self.ordered_parameters = list(self.parameters())


	def _create_autograd_function(self):
		class CustomAutogradFunction(autograd.Function):

			@staticmethod
			def forward(ctx, *x):
				self.previous_input = x[0]
				return self.child_forward(x[0])


			@staticmethod
			def backward(ctx, *grads):
				parameters, ordered_detached_params = self._attach_copied_parameters()

				with torch.enable_grad():
					y = grads[0].detach()
					layer_loss = torch.sum((self.child_forward(self.previous_input) - y) ** 2)
					layer_loss.requires_grad_()
				
				gradients = []
				for parameter in ordered_detached_params:
					gradients.append(torch.autograd.grad(layer_loss, parameter)[0])

				self._attach_original_parameters(parameters)

				return (self.inverse(y), *gradients)
		
		self.custom_forward = CustomAutogradFunction()


	def forward_override(self, x):
		if not self.custom_forward:
			self._attach_parameters()
			self._create_autograd_function()

		return self.custom_forward.apply(*([x] + self.ordered_parameters))


	def _attach_copied_parameters(self):
		parameters = {}
		ordered_detached_params = []
		for parameter in self.ordered_parameters:
			parameters[parameter.parameter_name] = parameter
			detached_param = nn.parameter.Parameter(parameter.clone().detach().requires_grad_())
			ordered_detached_params.append(detached_param)
			setattr(self, parameter.parameter_name, detached_param)
		return parameters, ordered_detached_params


	def _attach_original_parameters(self, parameters):
		for parameter in self.ordered_parameters:
			setattr(self, parameter.parameter_name, parameters[parameter.parameter_name])



	# TODO: generalize to multiple inputs
	def inverse(self, y):
		# e.g. returns x, ParameterDict({"W": W, "b", b})
		raise NotImplementedError("Each TargetPropagationLayer must implement inverse().")

