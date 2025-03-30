import pandas as pd
data=pd.read_csv('/content/slr.csv')
data.head()
X = data[['chirps per second']]
y = data['temperature (F)']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
test_size = 0.25)
print('Whole Data shape', data.shape)
print('X_train shape', X_train.shape)
print('X_test shape', X_test.shape)
from sklearn.linear_model import LinearRegression
slr_model = LinearRegression()
slr_model.fit(X_train, y_train)
#To see the simple linear regression parameters
slope = slr_model.coef_
y_intercept = slr_model.intercept_
print('Slope', slope)
print('Intercept', y_intercept)
y_pred_train = slr_model.predict(X_train)
y_pred_test = slr_model.predict(X_test)
Comparison_df = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred_test})
Comparison_df.to_csv(r'/content/Comparison_df.csv', index=True)
Comparison_df
