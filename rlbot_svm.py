import os, dotenv;
import numpy as np;
import pandas as pd;
from datetime import datetime, timedelta;

# To fetch data
import yfinance as yf;

# Need'em for that SVM
from sklearn.svm import SVC;
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score;
from sklearn.metrics import roc_auc_score, roc_curve, auc;

# Plotting ROC/AUC
import matplotlib.pyplot as plt;


#
# Fetch data sources
#

start_date 	= '2016-01-01';
end_date 	= '2023-01-01';

# 1762 Observations in DD, DJIA, DJTA, SPX, VIX but not W5000
dd = yf.Ticker('^DJI').history(start=start_date, end=end_date);
dd.index = pd.to_datetime(dd.index.date);

djia = yf.Ticker('^DJT').history(start=start_date, end=end_date);
djia.index = pd.to_datetime(djia.index.date);

djta = yf.Ticker('^DJT').history(start=start_date, end=end_date);
djta.index = pd.to_datetime(djta.index.date);

spx = yf.Ticker('^SPX').history(start=start_date, end=end_date);
spx.index = pd.to_datetime(spx.index.date);

wilshire = yf.Ticker('^W5000').history(start=start_date, end=end_date);
wilshire.index = pd.to_datetime(wilshire.index.date);

vix = yf.Ticker('^VIX').history(start=start_date, end=end_date);
vix.index = pd.to_datetime(vix.index.date);


# Wilshire5000 inexplicably contains 1756 observations.
# Filter other dataframes to match wilshire size.

wilshire_dates = list(wilshire.index.astype(str));

dd 		= dd[dd.index.astype(str).isin(wilshire_dates)];
djia 	= djia[djia.index.astype(str).isin(wilshire_dates)];
djta 	= djta[djta.index.astype(str).isin(wilshire_dates)];
spx 	= spx[spx.index.astype(str).isin(wilshire_dates)];
vix 	= vix[vix.index.astype(str).isin(wilshire_dates)];


#
# Now that every constituent dataframe contains 1756 rows, organize closing prices into a single dataframe
#

market_data = pd.DataFrame({
	'DD': dd['Close'],
	'DJIA': djia['Close'],
	'DJTA': djta['Close'],
	'SPX': spx['Close'],
	'W5000': wilshire['Close'],
	'VIX': vix['Close'],
});

# >>> market_data
#                       DD          DJIA          DJTA          SPX         W5000        VIX
# 2016-01-04  17148.939453   7352.589844   7352.589844  2012.660034  20841.359375  20.700001
# 2016-01-05  17158.660156   7363.950195   7363.950195  2016.709961  20879.929688  19.340000
# 2016-01-06  16906.509766   7217.049805   7217.049805  1990.260010  20593.400391  20.590000
# 2016-01-07  16514.099609   6995.390137   6995.390137  1943.089966  20089.529297  24.990000
# 2016-01-08  16346.450195   6946.359863   6946.359863  1922.030029  19867.099609  27.010000
# ...                  ...           ...           ...          ...           ...        ...
# 2022-12-23  33203.929688  13564.759766  13564.759766  3844.820068  38110.421875  20.870001
# 2022-12-27  33241.558594  13530.309570  13530.309570  3829.250000  37931.738281  21.650000
# 2022-12-28  32875.710938  13298.360352  13298.360352  3783.219971  37460.910156  22.139999
# 2022-12-29  33220.800781  13496.230469  13496.230469  3849.280029  38159.328125  21.440001
# 2022-12-30  33147.250000  13391.910156  13391.910156  3839.500000  38073.941406  21.670000
#
# [1756 rows x 6 columns]


#
# Convert the daily closing prices in each column into a percentage change with respect to the previous day.
#

market_data['DD_delta'] = (market_data['DD'].diff() > 0).astype(int);

# Chat recommended this but I don't believe it is useful/productive
# market_data['DD_delta'] = market_data['DD_delta'].shift(-1);

for col in ['DJIA', 'DJTA', 'SPX', 'W5000', 'VIX']:
	market_data[f'{col}_delta'] = market_data[col].pct_change();

market_data = market_data.dropna(); # drop first row after computing those percent changes because it has NaNs


#
# Apply leave-one-out cross-validation to partition the data into training and testing sets
#

X = market_data[['DJIA_delta', 'DJTA_delta', 'SPX_delta', 'W5000_delta', 'VIX_delta']];
y = market_data['DD_delta'];

loo_cv = LeaveOneOut();


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
# Fit SVM to training data with Polynomial Kernel Function
#

svm_polynom = SVC(kernel='poly', degree=3, probability=True);
svm_polynom_scores = cross_val_score(svm_polynom, X, y, cv=k_fold, scoring='roc_auc');


#
# Evaluate model's performance on testing data (repeat for polynom if the rbf is no good)
#

svm_rbf.fit(X,y);
y_predict = svm_rbf.predict_proba(X)[:,1];


#
# Compute FPR, TPR and ROC/AUC
# 


# >>> roc_auc
# 0.9372569354420535 <- Lets fucking goooooooooo
# This is a pretty good score, so we could probably incorporate this into the RL agent state space when it considers hold/buy/sell DuPont
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