import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CharEmbedding(nn.Module):
	def __init__(self, char_single_embedding_dim, char_embedding_dim, filter_height, dropout, char_vocab_size):
		super(CharEmbedding, self).__init__()
		self.char_single_embedding_dim = char_single_embedding_dim
		self.char_embedding_dim = char_embedding_dim
		self.filter_height = filter_height
		self.dropout = dropout
		self.char_vocab_size = char_vocab_size

		self.embedding_lookup = nn.Embedding(self.char_vocab_size, char_single_embedding_dim)
		self.cnn = nn.Conv1d(char_single_embedding_dim, char_embedding_dim, filter_height, padding=0)

	def forward(self, text_char):
		batch_size, max_length, max_word_length = text_char.size()

		# embedding look up
		text_char = text_char.contiguous().view(batch_size * max_length, max_word_length)
		text_char = self.embedding_lookup(text_char)
		assert text_char.size() == (batch_size * max_length, max_word_length, self.char_single_embedding_dim)
		
		# cnn
		text_char = torch.transpose(text_char, 1, 2)
		assert text_char.size() == (batch_size * max_length, self.char_single_embedding_dim, max_word_length)
 		text_char = self.cnn(text_char)
 		assert text_char.size() == (batch_size * max_length, self.char_embedding_dim, -1)
 		text_char = text_char.contiguous().view(batch_size * max_length * self.char_embedding_dim, -1)
 		text_char = nn.functional.relu(text_char)

 		# maxpool 
 		text_char = torch.max(text_char, 1)
 		assert text_char.size() == (batch_size * max_length * self.char_embedding_dim, 1)
 		text_char = text_char.contiguous().view(batch_size, max_length, self.char_embedding_dim)
 		return text_char
