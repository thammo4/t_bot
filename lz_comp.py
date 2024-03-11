import numpy as np;
import pandas as pd;
import yfinance as yf;



def binarize (data):
	b_seq = np.zeros_like(data, dtype=int);
	b_seq[1:] = (data[1:].values > data[:-1].values).astype(int);

	return b_seq;


def symbolize (data, alphabet_size=10):
	data_min = data.min();
	data_max = data.max();
	data_range = data_max - data_min;

	symbols = np.floor((data-data_min) / (data_range / alphabet_size)).astype(int);

	return symbols;


def lz_compression (sequence):
	dictionary = {};
	compressed = [];
	pattern = '';
	next_code = 1;

	for char in sequence:
		pattern_char = pattern + str(char);

		if pattern_char not in dictionary:
			compressed.append(dictionary.get(pattern,0));
			dictionary[pattern_char] = next_code;
			next_code += 1;
			pattern = str(char);
		else:
			pattern = pattern_char;

	compressed.append(dictionary.get(pattern,0));
	return compressed;



#
# Apply Lempel-Ziv compression to sequence of increase/decrease for dupont stock price
#

df = yf.Ticker('DD').history(period='5mo');
df_binary = binarize(df['Close']);
df_compress = lz_compression(df_binary);

print(f'Compression Ratio: {len(df_binary)/len(df_compress)}');
# >>> Compression Ratio: 2.861111111111111













# # def lz_complexity (sequence):
# def lz_compression (sequence):
# 	dictionary = {};
# 	complexity = 1;
# 	pattern = '';

# 	for char in sequence:
# 		pattern_plus_char = pattern + str(char);
# 		if pattern_plus_char not in dictionary:
# 			dictionary[pattern_plus_char] = complexity;
# 			complexity += 1;
# 			pattern = '';
# 		else:
# 			pattern = pattern_plus_char;

# 	return complexity;











