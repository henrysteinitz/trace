import torch
from torch.testing import assert_close
import unittest

from associative import AssociativeAttention


class TestAssociativeAttention(unittest.TestCase):

	def test_single_attention(self):
		module = AssociativeAttention()
		query = torch.Tensor([[3, 3, 4, 4, 0, 0, 0]])
		memory = torch.Tensor([[
			[3, 3, 4, 4, 7, 7, 7],
			[1, 2, 1, 0, 0, 2, 2],
			[10, 0, 1, 0, 4, -1, 1],
		]])

		# Beacuse the query most closely matches the first row 
		# in memory the module should approximatelty complete the first 
		# row.
		assert_close(
			module(query, memory), 
			torch.tensor([[[3.0, 3.0, 4.0, 4.0, 7.0, 7.0, 7.0]]]), 
			atol=.01, 
			rtol=0
		)


if __name__ == '__main__':
    unittest.main()