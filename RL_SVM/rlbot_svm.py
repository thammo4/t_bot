import os, dotenv;
import numpy as np;
import pandas as pd;
from datetime import datetime, timedelta;

# To fetch data
import yfinance as yf;

# Need'em for that SVM
from sklearn.svm import SVC;
from sklearn.preprocessing import StandardScaler;
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score;
from sklearn.metrics import roc_auc_score, roc_curve, auc;

# Plotting ROC/AUC
import matplotlib.pyplot as plt;


#
# Define custom kernel because financial data is a huge pain
#

# def rbf_linear_kernel (X,Y):
# 	X = X.to_numpy(); Y = Y.to_numpy();
# 	return np.dot(X,Y.T) + np.exp(-np.linalg.norm(X[:,np.newaxis]-Y,axis=2)**2);


#
# Fetch data sources
#

start_date 	= '2013-01-01';
end_date 	= '2023-11-01';

# 1762 Observations in DD, DJIA, DJTA, SPX, VIX but not W5000
dd = yf.Ticker('DD').history(start=start_date, end=end_date);
dd.index = pd.to_datetime(dd.index.date);

djia = yf.Ticker('^DJT').history(start=start_date, end=end_date);
djia.index = pd.to_datetime(djia.index.date);

djta = yf.Ticker('^DJT').history(start=start_date, end=end_date);
djta.index = pd.to_datetime(djta.index.date);

spx = yf.Ticker('^SPX').history(start=start_date, end=end_date);
spx.index = pd.to_datetime(spx.index.date);

wilshire = yf.Ticker('^W5000').history(start=start_date, end=end_date);
wilshire.index = pd.to_datetime(wilshire.index.date);

# vix = yf.Ticker('^VIX').history(start=start_date, end=end_date);
# vix.index = pd.to_datetime(vix.index.date);


# Wilshire5000 inexplicably contains 1756 observations.
# Filter other dataframes to match wilshire size.

wilshire_dates = list(wilshire.index.astype(str));

dd 		= dd[dd.index.astype(str).isin(wilshire_dates)];
djia 	= djia[djia.index.astype(str).isin(wilshire_dates)];
djta 	= djta[djta.index.astype(str).isin(wilshire_dates)];
spx 	= spx[spx.index.astype(str).isin(wilshire_dates)];
# vix 	= vix[vix.index.astype(str).isin(wilshire_dates)];


#
# Now that every constituent dataframe contains 1756 rows, organize closing prices into a single dataframe
#

market_data = pd.DataFrame({
	'DD': dd['Close'],
	'DJIA': djia['Close'],
	'DJTA': djta['Close'],
	'SPX': spx['Close'],
	'W5000': wilshire['Close'],
	# 'VIX': vix['Close'],
});

normalizer = StandardScaler();
market_data = pd.DataFrame(normalizer.fit_transform(market_data), columns=market_data.columns);

#
# Convert the daily closing prices in each column into a percentage change with respect to the previous day.
#

market_data['DD_delta'] = (market_data['DD'].diff() > 0).astype(int);

# Chat recommended this but I don't believe it is useful/productive
market_data['DD_delta'] = market_data['DD_delta'].shift(-1);

# for col in ['DJIA', 'DJTA', 'SPX', 'W5000', 'VIX']:
for col in ['DJIA', 'DJTA', 'SPX', 'W5000']:
	market_data[f'{col}_delta'] = market_data[col].pct_change();

market_data = market_data.dropna(); # drop first row after computing those percent changes because it has NaNs


#
# Apply leave-one-out cross-validation to partition the data into training and testing sets
#

# X = market_data[['DJIA_delta', 'DJTA_delta', 'SPX_delta', 'W5000_delta', 'VIX_delta']];
X = market_data[['DJIA_delta', 'DJTA_delta', 'SPX_delta', 'W5000_delta']];
y = market_data['DD_delta'];

# loo_cv = LeaveOneOut();


#
# Fit SVM to the training data using a Radial Basis Function Kernel
# LOO-CV (POSS) TAKES A REALLY LONG TIME TO RUN. CHAT ESTIMATES BETWEEN 5 MINS AND AROUND 5 HOURS
# Use the k-fold cross validation (k=10) unless you have much time to spare.
#

# svm_rbf_cv = cross_val_score(svm_rbf, X, y, cv=loo_cv, scoring='roc_auc');
k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=69);


svm_rbf = SVC(kernel='rbf', probability=True);
svm_rbf_scores = cross_val_score(svm_rbf, X, y, cv=k_fold, scoring='roc_auc');

#
# Compute FPR, TPR and ROC/AUC
# 

false_positive_rate, true_positive_rate, threshold = roc_curve(y, y_predict);
roc_auc = auc(false_positive_rate, true_positive_rate);


#
# Plot the ROC/AUC
#

plt.figure();
plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label='ROC Curve (area=%0.2f)' % roc_auc);
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate (FPR)');
plt.ylabel('True Positive Rate (TPR)');
plt.title('ROC for SVM Classifer of DuPont Closing Price Change with RBF Kernel');
plt.legend(loc='lower right');
plt.show();

