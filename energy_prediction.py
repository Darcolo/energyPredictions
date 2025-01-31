import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Steel_industry_data.csv")

data['Hour_of_Day'] = data['NSM'] // 3600
data['Minute_of_Hour'] = (data['NSM'] % 3600) // 60

def categorize_time(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

data['Time_of_Day'] = data['Hour_of_Day'].apply(categorize_time)


print(data.describe())
print(data.isnull().sum())
print(data.head())

# # Histogram
# plt.hist(data['Usage_kWh'], bins=30)
# plt.xlabel('Usage_kWh')
# plt.ylabel('Frequency')
# plt.title('Histogram of Energy Consumption')
# plt.show()

# # Scatter Plot
# plt.scatter(data['Lagging_Current_Reactive.Power_kVarh'], data['Usage_kWh'])
# plt.xlabel('Lagging_Current_Reactive.Power_kVarh')
# plt.ylabel('Usage_kWh')
# plt.title('Usage_kWh vs. Lagging_Current_Reactive.Power_kVarh')
# plt.show() 


X = data[['Lagging_Current_Reactive.Power_kVarh', 'Leading_Current_Reactive_Power_kVarh', 
          'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'Hour_of_Day','Minute_of_Hour']]  # Features
y = data['Usage_kWh']  # Target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#data preprocessing
scaler = StandardScaler()

scaler.fit(X_train)
scaler.fit(X_test)
x_train_scaled = scaler.transform(X_train)
x_test_scaled = scaler.transform(X_test)

# Training 
model = LinearRegression()
model.fit(x_train_scaled, y_train)
y_pred = model.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error = {mse}")
print(f"R Squared = {r2}")

# plt.scatter(y_test, color='red', label='Actual Values')
# plt.scatter(y_pred, color='blue', label='Predicted Values')
# plt.title('Comparison of Predicted and Actual Values')
# plt.legend()
# plt.show()
