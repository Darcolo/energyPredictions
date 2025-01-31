import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
 

data = pd.read_csv('Steel_industry_data.csv')

X = data[['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 
          'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor']]  # Features
y = data['Usage_kWh']  # Target 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

scaler.fit(x_train)
scaler.fit(x_test)
X_train_scaled = scaler.transform(x_train)
X_test_scaled = scaler.transform(x_test)


param_grid = {'n_neighbors': range(1, 21)}  # Test k values from 1 to 20
knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error') # 5-fold CV
grid_search.fit(X_train_scaled, y_train)

best_k = grid_search.best_params_['n_neighbors']
best_knn = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_knn.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)
print(f"Best k: {best_k}")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2_score}")