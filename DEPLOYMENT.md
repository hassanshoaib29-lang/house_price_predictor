# 🚀 Deployment Guide

## Recommended: Deploy to Streamlit Cloud

Streamlit Cloud is the best choice for this Python-based ML application because:

✅ **Native Python support** - No code conversion needed  
✅ **Built-in ML/DS support** - Great for scikit-learn, pandas, plotly  
✅ **Free hosting** - Generous free tier  
✅ **Easy deployment** - GitHub integration  
✅ **Automatic updates** - Re-deploys on push to main  

❌ Vercel would require converting to Next.js/React - **Not recommended for this project**

## Step-by-Step Deployment to Streamlit Cloud

### 1. Prepare Your GitHub Repository

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: House Price Predictor"

# Create main branch
git branch -M main

# Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/YOURUSERNAME/house-price-predictor.git

# Push to GitHub
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [https://share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Configure your app:
   - **Repository**: `YOURUSERNAME/house-price-predictor`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**

### 3. Access Your Deployed App

Your app will be available at:
```
https://YOURUSERNAME-house-price-predictor.streamlit.app
```

## Alternative: Deploy to Hugging Face Spaces

Hugging Face Spaces also supports Streamlit apps:

1. Create a new Space at [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as the SDK
3. Upload your code
4. Your app will be live at `https://YOURUSERNAME-house-price-predictor.hf.space`

## Testing Locally Before Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Run tests
python tests/test_models.py
```

## Troubleshooting Deployment Issues

### Common Issues

1. **Missing dependencies**
   - Ensure `requirements.txt` has all necessary packages
   - Check versions are compatible

2. **Data file not found**
   - Use relative paths in code
   - Verify `data/house_data.csv` exists

3. **Memory issues**
   - Reduce model complexity (fewer estimators)
   - Use `@st.cache_resource` to cache loaded models

### Local Development Tips

```bash
# Clear Streamlit cache
streamlit cache clear

# Run in debug mode
streamlit run app.py --logger.level=debug

# Check for config issues
streamlit config show
```

## Custom Domain (Optional)

For Streamlit Cloud:
1. Go to your app settings
2. Add custom domain
3. Configure DNS records

## Monitoring & Analytics

Add Streamlit's built-in analytics:
```python
import streamlit as st

st.set_page_config(page_title="House Price Predictor")
# Analytics is automatically enabled on Streamlit Cloud
```

## Security Considerations

- ✅ No sensitive data in the app
- ✅ Models are public
- ✅ No user authentication needed
- ✅ Read-only data access

## Cost Summary

| Platform | Cost | Notes |
|----------|------|-------|
| Streamlit Cloud | **Free** | 3 apps, 1GB RAM |
| Hugging Face Spaces | **Free** | Unlimited public spaces |
| Heroku | $7/month | More control |
| AWS/GCP | Pay-as-you-go | Full control |

---

**Recommendation: Use Streamlit Cloud** - It's free, easy, and optimized for exactly this type of application.
