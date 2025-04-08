# Deploying to Streamlit Cloud

This guide will help you deploy the Medical Image Super-Resolution application to Streamlit Cloud.

## Prerequisites

1. A GitHub account
2. Your code pushed to a GitHub repository
3. A Streamlit Cloud account (sign up at https://streamlit.io/cloud)

## Deployment Steps

1. **Push your code to GitHub**
   - Make sure all the files we created are in your repository:
     - `streamlit_app.py`
     - `requirements-streamlit.txt`
     - `.streamlit/config.toml`
     - `setup.sh`

2. **Sign in to Streamlit Cloud**
   - Go to https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Create a new app**
   - Click "New app"
   - Select your repository
   - Set the main file path to `streamlit_app.py`
   - Set the Python requirements file to `requirements-streamlit.txt`
   - Choose the branch (usually `main` or `master`)
   - Click "Deploy"

4. **Advanced settings (optional)**
   - If your models are large, you may need to select a higher resource tier
   - You can set environment variables if needed

## Troubleshooting

If you encounter issues during deployment:

1. **Check the logs**
   - Streamlit Cloud provides logs that can help identify issues

2. **Common issues**
   - Missing dependencies: Make sure all required packages are in `requirements-streamlit.txt`
   - File not found errors: Check that all paths in your code are correct
   - Memory issues: Try reducing model size or optimizing your code

3. **Model loading**
   - If models are too large for Streamlit Cloud, consider:
     - Using smaller models
     - Hosting models elsewhere and downloading at runtime
     - Using model quantization to reduce size

## Additional Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Components](https://streamlit.io/components)
- [Streamlit Forum](https://discuss.streamlit.io/)
