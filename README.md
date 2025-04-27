# Graduate Admission Prediction using ANN

This project predicts the likelihood of a student’s admission to a graduate program based on several features such as GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, and Research experience. An Artificial Neural Network (ANN) model has been built for accurate prediction.

## 📚 Dataset
- **Name**: Graduate Admissions Dataset
- **Source**: Kaggle (https://www.kaggle.com/datasets/mohansacharya/graduate-admissions)
- **Features**:
  - GRE Score (out of 340)
  - TOEFL Score (out of 120)
  - University Rating (1 to 5)
  - Statement of Purpose (SOP) Strength (1 to 5)
  - Letter of Recommendation (LOR) Strength (1 to 5)
  - Undergraduate GPA (CGPA) (out of 10)
  - Research Experience (0 or 1)
- **Target**:
  - Chance of Admit (ranging from 0 to 1)

## 🛠️ Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## 🧠 Model Architecture
- Input Layer: 7 Neurons (for 7 features)
- Hidden Layers: 2 Dense Layers
  - First Hidden Layer: 64 neurons (ReLU activation)
  - Second Hidden Layer: 32 neurons (ReLU activation)
- Output Layer: 1 Neuron (Sigmoid activation for prediction between 0 and 1)

## ⚙️ How to Run
1. **Clone the repository**  
   ```bash
   git clone <repository_link>
   cd graduate-admission-prediction
   ```

2. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Project**  
   ```bash
   python main.py
   ```

## 📈 Results
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Training R² Score**: **0.8011**
- The model shows strong prediction performance and generalizes well to unseen data.

## 📊 Visualizations
- **Training and Validation Loss Curves**: Line plot showing loss reduction over epochs.

## 🚀 Future Improvements
- Hyperparameter tuning (epochs, batch size, number of neurons)
- Add more visualizations (correlation heatmaps, actual vs predicted plots)
- Try other models like Random Forest, XGBoost for comparison
- Deploy the model using a simple Flask or Streamlit app

## 🧑‍💻 Author
- **Name**: [Debshuvra Sarkar]
- **University**: Jadavpur University
- **Department**: Power Engineering
- **Year**: 1st Year

---

⭐ If you like the project, feel free to star it and suggest improvements!
