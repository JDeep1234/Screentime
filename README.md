# 🖥️ Comprehensive Screen Time Analysis Tool

## 📝 Project Overview

This Streamlit-based application provides a comprehensive analysis of screen time data, offering advanced data visualization, machine learning insights, and personalized recommendations. The tool helps users understand the impact of screen time on various aspects of life through interactive and intelligent analysis.

## ✨ Features

### 1. Data Visualization Dashboard
- **Correlation Heatmap**: Visualize relationships between numeric features
- **Distribution Analysis**: Understand data distribution and outliers
- **Scatter Plots**: Explore relationships between variables
- **Box Plots**: Compare numeric variables across categorical groups
- **Pair Plot**: Comprehensive multi-variable visualization

### 2. Machine Learning Analysis
- Automatic problem type detection (Classification/Regression)
- Multiple model training:
  - XGBoost
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - Random Forest
- Performance metrics and comparison
- Feature importance analysis
- Cross-validation scoring

### 3. Contextual Recommendations
- Personalized insights based on data characteristics
- Recommendations for:
  - Screen Time Management
  - Mental Health
  - Physical Activity

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip (Python Package Manager)

### Clone the Repository
```bash
git clone https://github.com/yourusername/screen-time-analysis.git
cd screen-time-analysis
```

### Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Create requirements.txt
```
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
plotly
```

## 🖥️ Running the Application
```bash
streamlit run screentime_analyzer.py
```

## 📊 How to Use

1. Upload a CSV file with screen time and related data
2. Navigate through different analysis types:
   - Data Visualization
   - Machine Learning Analysis
   - Comprehensive Report

### Sample CSV Structure
Recommended columns:
- Screen Time
- Mental Health Score
- Physical Activity
- Other numeric/categorical variables

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🛠️ Troubleshooting

- Ensure all dependencies are installed
- Check Python version compatibility
- Verify CSV file format and structure

## 📈 Future Roadmap
- [ ] Add more advanced machine learning models
- [ ] Implement more sophisticated recommendation system
- [ ] Create interactive data cleaning tools
- [ ] Add more visualization options

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 🏆 Acknowledgements
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- XGBoost

