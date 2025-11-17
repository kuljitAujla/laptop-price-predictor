# Laptop Price Predictor

A machine learning project exploring linear regression and neural networks to predict laptop prices based on technical specifications. This project was developed as a learning exercise to understand AI/ML concepts with Python and to experiment with different regression techniques.

## Project Goals

- Learn and apply AI/ML concepts using Python
- Explore the effectiveness of linear regression for price prediction
- Experiment with neural networks for regression tasks
- Gain hands-on experience with data preprocessing and feature engineering

## Dataset

The dataset used in this project is the **Laptop Price Dataset** from Kaggle:
- **Source**: [Kaggle - Laptop Price Dataset](https://www.kaggle.com/datasets/ironwolf437/laptop-price-dataset)
- **Size**: ~1,275 laptop entries (with there was more. I feel the lack of data caused some issues)
- **Features**: Company, product type, screen specifications, CPU details, RAM, memory, GPU, weight, and price (in Euros)

## Data Preprocessing


### Screen Resolution Processing
- Extracted X and Y resolution values
- Calculated Pixels Per Inch (PPI)
- Identified touchscreen capabilities

### Memory Processing
- Separated SSD and HDD storage
- Converted TB to GB for consistency
- Created total memory feature

### Feature Engineering
- Extracted CPU series from CPU type
- Cleaned GPU information
- Removed features with weak correlation to price
- Applied one-hot encoding for categorical variables

### Final Features Used
- **Numerical**: RAM, SSD (GB), screen resolution (x_res, y_res), PPI, CPU speed
- **Categorical**: Company, type (Gaming, Ultrabook, Notebook, etc.), CPU company, GPU company, CPU series
- **Derived**: Touchscreen (binary), total memory

## Models Implemented

### 1. Simple Linear Regression (sklearn)
- **Library**: `sklearn.linear_model.LinearRegression`
- **Performance**:
  - R² Score: **0.771**
  - RMSE: **262.17**
  - MAE: **181.36**

### 2. Neural Network Regression (TensorFlow/Keras)
- **Architecture**: 
  - Dense layer (64 units) + Batch Normalization + ReLU
  - Dense layer (32 units) + ReLU
  - Output layer (1 unit)
- **Optimizer**: Adam (learning rate: 0.001)
- **Training**: 200 epochs with validation data
- **Data Scaling**: StandardScaler from sklearn
- **Performance**:
  - R² Score: **0.861**
  - RMSE: **187.12**
  - MAE: **132.73**

### Results Comparison

The neural network model outperformed the simple linear regression model, achieving:
- **4.2% higher R² score** (0.793 vs 0.761)
- **7.0% lower RMSE** (243.93 vs 262.17)
- **3.7% lower MAE** (174.64 vs 181.36)

While both models showed good predictive capabilities, the neural network's ability to capture non-linear relationships provided a modest but meaningful improvement.

### Future Models
- This was a learning project and I am eager to try new models like randomForests and more.
- However, that is something I will come back to as now I am taking a break from supervised learning and trying unsupervised Learning


## Getting Started

### Prerequisites

- Python 3.7+ (Ideally use Anaconda base environment and install dependancies from there)
- Jupyter Notebook
- Required Python packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - imbalanced-learn (imblearn)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd laptop-price-predictor
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imbalanced-learn
```

3. Open the Jupyter notebook:
```bash
jupyter notebook laptop-price-regression.ipynb
```

## Key Insights

1. **Most Important Features** for price prediction:
   - RAM (correlation: 0.725)
   - SSD capacity (correlation: 0.612)
   - Core i7 CPU series (correlation: 0.584)
   - Screen resolution (correlation: ~0.483)

2. **Neural Networks** can capture more complex patterns than simple linear regression, though the improvement was modest in this case. Probably because I'm still quite new to this and also didn't do as aggressive and effecient hyperparameter tuning as the previous project (also in github)

3. **Data preprocessing** was crucial - feature engineering and removing weak/leaking features significantly improved model performance.

## Lessons Learned

- Linear regression provides a solid baseline and is highly interpretable. Also very easy to implement XD
- Neural networks require more hyperparameter tuning and can be tedious to optimize
- Feature engineering often has a greater impact on performance than model complexity
- Data leakage (like GPU model numbers that indirectly encode price) must be carefully identified and removed
- Standardization is essential for neural network training

## License

This project is for educational purposes. The dataset is sourced from Kaggle and should be used in accordance with its original license.

## Acknowledgments

- Dataset: [IRON WOLF. (2024). Laptop Price - dataset](https://www.kaggle.com/datasets/ironwolf437/laptop-price-dataset)
- Built with Python, scikit-learn, and TensorFlow
