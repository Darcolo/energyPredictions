import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Steel_industry_data.csv')

# Data Preprocessing
X = data[['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 
          'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor']]  # Features
y = data['Usage_kWh']  # Target 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scale = MinMaxScaler()
scale.fit(x_train)
scale.fit(x_test)
x_train_scaled = scale.transform(x_train)
x_test_scaled = scale.transform(x_test)


# Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(x_train_scaled, y_train)
y_pred = tree_model.predict(x_test_scaled)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
