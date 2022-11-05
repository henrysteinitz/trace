import torch
from torch.testing import assert_close
import unittest

from contrastive import ContrastiveAttention


class TestContrastiveAttention(unittest.TestCase):

	def test_single_attention(self):
		module = ContrastiveAttention()
		query = torch.Tensor([[3, 3, 4, 4, 7, 7, 7]])
		memory = torch.Tensor([[
			[3, 3, 4, 4, 7, 0, 0],
			[3, 3, 4, 0, 0, 0, 7],
			[3, 3, 0, 4, 0, 7, 7],
			[-3, 9, 1, -8, 0, 2, 2],
		]])

		# Beacuse the query is very similar like to the first three entries, 
		# the module should aprroximately compute the last entry.
		print(module(query, memory))
		assert_close(
			module(query, memory), 
			torch.tensor([[[-3.0, 9.0, 1.0, -8.0, 0, 2.0, 2.0]]]), 
			atol=.01, 
			rtol=0
		)

if __name__ == '__main__':
    unittest.main()