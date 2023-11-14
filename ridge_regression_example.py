#
# RIDGE Regression Example (for temporary reference during development)
# 	• NB: for python, `alpha` is the ordinary lambda tuning parameter
#

df = pd.read_csv("https://raw.githubusercontent.com/Statology/Python-Guides/main/mtcars.csv")[['mpg', 'wt', 'drat', 'qsec', 'hp']];

# Define covariates (X) and response (Y)
# For stock trading:
# 	• df_X = Factors that contribute to / affect the closing price of a stock
# 	• df_Y = Closing price of a stock
df_X = df[['mpg', 'wt', 'drat', 'qsec']];
df_Y = df['hp'];

# Perform 10-fold cross validation three times
# For stock trading:
# 	• This can probably stay the same
xval = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1);

# Define model and fit to training data
# For stock trading:
# 	• This can probably stay the same
model = RidgeCV(alphas=np.arange(.01, 1, .01), cv=xval, scoring='neg_mean_absolute_error');
model.fit(df_X, df_Y);

print('Tuning parameter (lambda) corresponding to lowest test MSE: {}'.format(model.alpha_));

# Prediction with RIDGE Model
# For stock trading:
# 	• Update the df_new to contain the future values of the covariates from df_X
df_new = pd.DataFrame({'mpg':[24], 'wt':[2.5], 'drat':[3.5], 'qsec':[18.5]});
df_predicted = model.predict(df_new);

print(f'Predicted HP: {df_predicted[0]}');