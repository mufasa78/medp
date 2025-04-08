# Medical Image Super-Resolution - Streamlit App

This is a Streamlit version of the Medical Image Super-Resolution application, designed for deployment on Streamlit Cloud.

## Features

- Single image super-resolution processing
- Batch processing of multiple images
- Multiple deep learning models (SRCNN, ESPCN, EDSR, RCAN, SRResNet)
- Classical super-resolution methods for comparison
- Image quality metrics (PSNR, SSIM)
- Bilingual interface (English and Chinese)

## Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app and select your forked repository
4. Set the main file path to `streamlit_app.py`
5. Deploy the app

## Local Development

To run the app locally:

1. Install the required dependencies:
   ```
   pip install -r requirements-streamlit.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Project Structure

- `streamlit_app.py`: Main Streamlit application
- `requirements-streamlit.txt`: Dependencies for Streamlit Cloud
- `.streamlit/config.toml`: Streamlit configuration
- `setup.sh`: Setup script for Streamlit Cloud
- `Procfile`: Process file for deployment

## Models

The application uses several deep learning models for super-resolution:

- SRCNN: Super-Resolution Convolutional Neural Network
- ESPCN: Efficient Sub-Pixel Convolutional Neural Network
- EDSR: Enhanced Deep Super-Resolution Network
- RCAN: Residual Channel Attention Network
- SRResNet: Super-Resolution Residual Network

Each model is available in both grayscale and color versions.

## Classical Methods

For comparison, the application also includes classical super-resolution methods:

- Bicubic Interpolation
- Iterative Back Projection (IBP)
- Non-Local Means (NLMeans)
- Edge-Guided Super-Resolution

## Image Quality Metrics

The application evaluates the quality of super-resolved images using:

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)

## Languages

The application supports both English and Chinese interfaces.
