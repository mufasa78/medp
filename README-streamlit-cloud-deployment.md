# Medical Image Super-Resolution - Streamlit Cloud Deployment Guide

This guide provides instructions for deploying the Medical Image Super-Resolution application to Streamlit Cloud.

## Deployment Steps

1. **Set up your GitHub repository**
   - Make sure all the necessary files are in your repository:
     - `streamlit_app.py` - The main Streamlit application
     - `requirements-streamlit.txt` - Dependencies for Streamlit Cloud
     - `.streamlit/config.toml` - Streamlit configuration
     - `setup.sh` - Setup script for Streamlit Cloud
     - All source code in the `src/` directory

2. **Deploy to Streamlit Cloud**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to `streamlit_app.py`
   - Set the Python requirements file to `requirements-streamlit.txt`
   - Click "Deploy"

3. **Set Environment Variables (Optional)**
   - After deployment, go to your app settings
   - Add the environment variable `IS_CLOUD_DEPLOYMENT` with value `true`
   - This will enable cloud-specific optimizations

## Deployment Optimizations

The application includes several optimizations for cloud deployment:

1. **Reduced Model Set**
   - When `IS_CLOUD_DEPLOYMENT=true`, only a subset of models will be loaded to save memory
   - The most efficient models (SRCNN, ESPCN) are prioritized

2. **Temporary File Storage**
   - In cloud mode, files are stored in temporary directories
   - This is compatible with Streamlit Cloud's read-only file system

3. **Graceful Fallbacks**
   - The app will work even if model checkpoints are not available
   - Missing dependencies are handled gracefully

4. **Memory Management**
   - GPU memory is cleared after model loading
   - Only necessary models are loaded

## Troubleshooting

If you encounter issues during deployment:

1. **Memory Errors**
   - Try setting `IS_CLOUD_DEPLOYMENT=true` to use fewer models
   - Consider removing some of the larger models from your repository

2. **Missing Dependencies**
   - Check the logs for missing packages
   - Update `requirements-streamlit.txt` as needed

3. **File Access Issues**
   - Make sure the app is using temporary directories in cloud mode
   - Check that all file paths use the `UPLOAD_DIR` and `RESULTS_DIR` variables

4. **Model Loading Errors**
   - Ensure model checkpoints are included in your repository
   - Or set `IS_CLOUD_DEPLOYMENT=true` to use untrained models if checkpoints are missing

## Testing Your Deployment

After deployment, test the following:

1. **Single Image Processing**
   - Upload a test image
   - Try different models
   - Check that results are displayed correctly

2. **Batch Processing**
   - Upload multiple images
   - Verify that all images are processed
   - Check that results are displayed in the gallery

3. **Language Switching**
   - Test switching between English and Chinese
   - Verify that all UI elements are translated

## Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Community Forum](https://discuss.streamlit.io/)
