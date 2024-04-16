import numpy as np;
import pandas as pd;


#
# Present Value Function - Compound Interest
#

def pv (fv, i, t):
	present_value = fv / np.power(1+i, t);
	return present_value;


#
# Future Value Function - Compound Interest
#

def fv (pv, i, t):
	future_value = pv * np.power(1+i, t);
	return future_value;