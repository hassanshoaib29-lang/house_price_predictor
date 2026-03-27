# 🏠 House Price Predictor

An interactive machine learning application that predicts house prices using multiple regression techniques. Built with Python, scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)

## 🚀 Live Demo

**Streamlit Cloud:** [Add your deployed URL here]

## 📊 Features

- **Multiple Regression Models**: Choose between Linear Regression, Random Forest, and Gradient Boosting
- **Interactive Interface**: Adjustable inputs for house features
- **Real-time Predictions**: Get instant price estimates
- **Model Performance Metrics**: View R² Score, RMSE, and MAE
- **Feature Importance Visualization**: See which factors impact prices most
- **Prediction History**: Track your recent searches
- **Beautiful UI**: Modern, responsive design

## 🏗️ House Features Analyzed

| Feature | Description | Range |
|---------|-------------|-------|
| Square Footage | Total living area | 100 - 10,000 sqft |
| Bedrooms | Number of bedrooms | 1 - 6+ |
| Bathrooms | Number of bathrooms | 1 - 4+ |
| Year Built | Construction year | 1800 - 2026 |
| Lot Size | Land area | 1,000 - 50,000 sqft |
| Garage | Car capacity | 0 - 3+ |
| Neighborhood Score | Area quality (1-10) | 1 - 10 |
| Location Type | Urban, Suburban, Rural | 1 - 3 |

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Clone the Repository

```bash
git clone https://github.com/yourusername/house-price-predictor.git
cd house-price-predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

**Streamlit (Recommended for Web):**
```bash
streamlit run app.py
```

**Tkinter (Desktop):**
```bash
python house_price_predictor.py
```

## 📁 Project Structure

```
house-price-predictor/
├── app.py                      # Streamlit web application
├── house_price_predictor.py    # Desktop Tkinter application
├── requirements.txt            # Python dependencies
├── data/
│   └── house_data.csv          # Training dataset
├── models/                     # Trained model files
├── notebooks/
│   └── model_training.ipynb    # Model development notebook
├── tests/
│   └── test_models.py          # Unit tests
├── .gitignore                  # Git ignore file
├── LICENSE                     # MIT License
├── README.md                   # This file
└── SPEC.md                     # Technical specification
```

## 🔬 Model Performance

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Random Forest | 0.997 | $9,622 | $7,234 |
| Gradient Boosting | 0.994 | $12,607 | $9,456 |
| Linear Regression | 0.990 | $16,717 | $12,890 |

## 🌐 Deployment

### Deploy to Streamlit Cloud (Recommended)

1. **Create GitHub Repository**
   - Push your code to GitHub

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository, branch, and main file (`app.py`)
   - Click "Deploy!"

3. **Custom Domain** (Optional)
   - Streamlit Cloud supports custom domains

### Deploy to Vercel

For Vercel, you would need to convert this to a Next.js application, which requires significant changes. **Streamlit is the better choice** for this Python-based ML application.

### Deploy to Heroku

```bash
# Install Heroku CLI
# Create Procfile with: web: streamlit run app.py --server.port $PORT
heroku create your-app-name
git push heroku main
heroku open
```

## 📖 Usage

### Using the Web Application

1. **Select Model**: Choose your preferred regression algorithm
2. **Enter House Details**: Fill in all house features
3. **Get Prediction**: Click "Predict" to see estimated price
4. **View Metrics**: Check model performance indicators
5. **Compare**: Try different models for comparison

### Example Prediction

```
Input:
- Square Footage: 2,000
- Bedrooms: 3
- Bathrooms: 2
- Year Built: 2000
- Lot Size: 8,000
- Garage: 2 cars
- Neighborhood Score: 7
- Location: Suburban

Output:
- Predicted Price: ~$289,500
- Price Range: $260,550 - $318,450
- Confidence: High (99.7% R²)
```

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Run specific test file
python tests/test_models.py
```

## 🔒 Environment Variables

If needed, create a `.env` file:

```env
MODEL_PATH=models/trained_model.pkl
DATA_PATH=data/house_data.csv
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset inspired by real estate market data
- Built with scikit-learn, Streamlit, and pandas
- Inspired by various housing price prediction projects

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/house-price-predictor](https://github.com/yourusername/house-price-predictor)

---

⭐ If you found this project useful, please give it a star!

