# Step 1: Load the dataset
import pandas as pd

df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

# Drop unnecessary column
df.drop(columns=['Serial No.'], inplace=True)

# Define features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Step 2: Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 3: Scale the features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Build the ANN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(7, activation='relu', input_dim=7),
    Dense(7, activation='relu'),
    Dense(1, activation='linear')
])

# Step 5: Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Step 6: Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

# Step 7: Make predictions and evaluate
y_pred = model.predict(X_test_scaled)

from sklearn.metrics import r2_score
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Step 8: Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Step 9: Correlation heatmap
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Step 10: Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.xlabel('Actual Chance of Admit')
plt.ylabel('Predicted Chance of Admit')
plt.title('Actual vs Predicted')
plt.plot([0, 1], [0, 1], 'r--')
plt.grid(True)
plt.show()

# Step 11: Save the model and scaler
model.save("admission_model.h5")
import joblib
joblib.dump(scaler, "scaler.save")
