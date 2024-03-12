import numpy as np;
import pandas as pd;


#
# Define hamming(7,4) encoder
#

def encode_hamming74 (data_bits):
	# assert len(data_bits) == 4, "Input must be nibble";
	if len(data_bits) != 4:
		print('hello world');

	d1, d2, d3, d4 = data_bits;

	p1 = d1^d2^d4;
	p2 = d1^d3^d4;
	p3 = d2^d3^d4;

	return [p1, p2, d1, p3, d2, d3, d4];



#
# Define hamming(7,4) decoder
#

def decode_hamming74 (encoded_bits):
	if len(encoded_bits) != 7:
		print('wtf');
		return;

	p1,p2,d1,p3,d2,d3,d4 = encoded_bits;


	p1_check = p1^d1^d2^d4;
	p2_check = p2^d1^d3^d4;
	p3_check = p3^d2^d3^d4;

	error_check = 1*p1_check + 2*p2_check + 4*p3_check;

	if error_check:
		print(f'ERROR AT {error_check-1}');
		encoded_bits[error_check-1] ^= 1;

	return [encoded_bits[2], encoded_bits[4], encoded_bits[5], encoded_bits[6]];



#
# Test
# • encode a 4 bit message
# • flip a bit to simulate an error
# • pass to decoder for error detection + correction
#

data = [1,1,0,1];
data_encode = encode_hamming74(data);

data_encode[6] ^= 1;

data_decode = decode_hamming74(data_encode);

print(f'Original: {data}');
print(f'Encoded: {data_encode}');
print(f'Decoded: {data_decode}');


#
# Example Output
#

# >>> python3 hamming74.py
# ERROR AT 6
# Original: [1, 1, 0, 1]
# Encoded: [1, 0, 1, 0, 1, 0, 1]
# Decoded: [1, 1, 0, 1]