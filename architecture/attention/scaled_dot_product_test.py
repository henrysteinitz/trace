import torch
from torch.testing import assert_close
import unittest

from scaled_dot_product import ScaledDotProductAttention


class TestScaledDotProductAttention(unittest.TestCase):

	def test_single_attention(self):
		module = ScaledDotProductAttention()
		query = torch.Tensor([[0, 3, 0, 0]])
		keys = torch.Tensor([[
			[0, 3, 0, 0],
			[0, 0, 0, 0],
			[0, 0, 0, 3]
		]])
		values = torch.Tensor([[
			[1, 2, 3],
			[9, 8, 7],
			[1, 8, 3]
		]])

		# Beacuse the query exactly matches the first key, 
		# the module should approximatelty compute the first 
		# value.
		assert_close(module(query, keys, values), torch.tensor([[[1.0, 2.0, 3.0]]]), atol=.2, rtol=0)


if __name__ == '__main__':
    unittest.main()