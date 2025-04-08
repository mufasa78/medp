import os
import torch
import streamlit as st # type: ignore
from PIL import Image
import io
import sys
import time
import numpy as np
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Handle potential import errors gracefully
try:
    from streamlit_image_comparison import image_comparison # type: ignore
    HAS_IMAGE_COMPARISON = True
except ImportError:
    logger.warning("streamlit_image_comparison package is not installed. Will use fallback display method.")
    HAS_IMAGE_COMPARISON = False
    image_comparison = None

# Detect deployment environment
IS_CLOUD_DEPLOYMENT = os.environ.get('IS_CLOUD_DEPLOYMENT', 'false').lower() == 'true'

# Set up file paths for uploads and results
if IS_CLOUD_DEPLOYMENT:
    # Use temporary directory for cloud deployment
    UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'uploads')
    RESULTS_DIR = os.path.join(tempfile.gettempdir(), 'results')
    logger.info(f"Cloud deployment detected. Using temporary directories: {UPLOAD_DIR}, {RESULTS_DIR}")
else:
    # Use static directory for local development
    UPLOAD_DIR = 'static/uploads'
    RESULTS_DIR = 'static/results'
    logger.info(f"Local deployment detected. Using static directories: {UPLOAD_DIR}, {RESULTS_DIR}")

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs('src/checkpoints', exist_ok=True)

# Add a message about checkpoints directory
if IS_CLOUD_DEPLOYMENT:
    logger.info("Cloud deployment: Model checkpoints will be loaded if available, otherwise untrained models will be used.")

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import SRCNN, ESPCN, EDSR, RCAN, SRResNet
from src.color_models import ColorSRCNN, ColorESPCN, ColorEDSR, ColorRCAN, ColorSRResNet
from src.classical_sr import BicubicInterpolation, IBP, NLMeans, EdgeGuidedSR
from src.datasets import BatchProcessor
from src.metrics import evaluate_image_quality
from src.translations import get_text
from torchvision.transforms import ToTensor, ToPILImage

# Set page configuration
st.set_page_config(
    page_title="Medical Image Super-Resolution",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Model configurations
if IS_CLOUD_DEPLOYMENT:
    # Use a reduced set of models for cloud deployment to save memory
    MODEL_CONFIGS = {
        'srcnn': {'class': SRCNN, 'checkpoint': 'src/checkpoints/srcnn_best.pth'},
        'espcn': {'class': ESPCN, 'checkpoint': 'src/checkpoints/espcn_best.pth'},
        'color_srcnn': {'class': ColorSRCNN, 'checkpoint': 'src/checkpoints/color_srcnn_best.pth'}
    }
    logger.info("Using reduced model set for cloud deployment")
else:
    # Use all models for local deployment
    MODEL_CONFIGS = {
        'srcnn': {'class': SRCNN, 'checkpoint': 'src/checkpoints/srcnn_best.pth'},
        'espcn': {'class': ESPCN, 'checkpoint': 'src/checkpoints/espcn_best.pth'},
        'edsr': {'class': EDSR, 'checkpoint': 'src/checkpoints/edsr_best.pth'},
        'rcan': {'class': RCAN, 'checkpoint': 'src/checkpoints/rcan_best.pth'},
        'srresnet': {'class': SRResNet, 'checkpoint': 'src/checkpoints/srresnet_best.pth'},
        'color_srcnn': {'class': ColorSRCNN, 'checkpoint': 'src/checkpoints/color_srcnn_best.pth'},
        'color_espcn': {'class': ColorESPCN, 'checkpoint': 'src/checkpoints/color_espcn_best.pth'},
        'color_edsr': {'class': ColorEDSR, 'checkpoint': 'src/checkpoints/color_edsr_best.pth'},
        'color_rcan': {'class': ColorRCAN, 'checkpoint': 'src/checkpoints/color_rcan_best.pth'},
        'color_srresnet': {'class': ColorSRResNet, 'checkpoint': 'src/checkpoints/color_srresnet_best.pth'}
    }

# Classical method configurations
CLASSICAL_METHODS = {
    'bicubic': BicubicInterpolation,
    'ibp': IBP,
    'nlmeans': NLMeans,
    'edge': EdgeGuidedSR
}

# Global variables for models
models = {}
classical_methods = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_processor = None

# Session state initialization
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'processed_images' not in st.session_state:
    st.session_state.processed_images = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

@st.cache_resource(show_spinner=False)
def load_model(model_class, model_name, checkpoint_path, scale_factor=2):
    """Load a single model with caching"""
    try:
        logger.info(f"Loading model: {model_name}")

        # Initialize model
        if model_name in ['espcn', 'edsr', 'rcan', 'srresnet', 'color_espcn', 'color_edsr', 'color_rcan', 'color_srresnet']:
            model = model_class(scale_factor=scale_factor).to(device)
        else:
            model = model_class().to(device)

        # Look for model checkpoint
        checkpoint_exists = os.path.exists(checkpoint_path)

        # For cloud deployment, we'll use untrained models if checkpoints don't exist
        # This allows the app to run even without the model files
        if checkpoint_exists:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Successfully loaded {model_name} from {checkpoint_path}")
                load_status = "success"
            except Exception as e:
                logger.warning(f"Error loading checkpoint for {model_name}: {e}. Using untrained model.")
                load_status = "warning"
        else:
            logger.warning(f"No checkpoint found for {model_name} at {checkpoint_path}, using untrained model")
            load_status = "warning"

        # Set model to evaluation mode to disable training behavior
        model.eval()

        # Free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model, load_status
    except Exception as e:
        logger.error(f"Error loading {model_name}: {e}")
        return None, "error"

def load_models():
    """Load all models and classical methods"""
    global models, classical_methods, batch_processor

    st.info(f"Loading models on device: {device}")

    # Create checkpoints directory if it doesn't exist
    os.makedirs('src/checkpoints', exist_ok=True)

    # Track loaded models for progress bar
    total_models = len(MODEL_CONFIGS)
    progress_bar = st.progress(0)
    status_container = st.empty()

    # Load deep learning models
    for i, (model_name, config) in enumerate(MODEL_CONFIGS.items()):
        # Update progress
        progress_bar.progress((i + 1) / total_models)
        status_container.text(f"Loading {model_name}...")

        # Load model with caching
        model, status = load_model(
            model_class=config['class'],
            model_name=model_name,
            checkpoint_path=config['checkpoint']
        )

        if model is not None:
            models[model_name] = model
            if status == "success":
                st.success(f"✅ Loaded {model_name}")
            elif status == "warning":
                st.warning(f"⚠️ Loaded {model_name} (untrained)")
        else:
            st.error(f"❌ Failed to load {model_name}")

    # Initialize classical methods
    for method_name, method_class in CLASSICAL_METHODS.items():
        try:
            classical_methods[method_name] = method_class(scale_factor=2)
            st.success(f"Initialized classical method: {method_name}")
        except Exception as e:
            st.error(f"Error initializing {method_name}: {e}")

    # Initialize batch processor with default model (SRCNN if available, otherwise first available)
    default_model = models.get('srcnn', next(iter(models.values())) if models else None)
    if default_model:
        batch_processor = BatchProcessor(
            model=default_model,
            device=device,
            batch_size=4,
            scale_factor=2,
            save_dir=RESULTS_DIR,
            preserve_color=False
        )
        st.success("Initialized batch processor")

def process_image(image, model_name='srcnn', classical_method=None, preserve_color=False):
    """Process image through the super-resolution model and/or classical method"""
    try:
        original_image = image.copy()

        # Resize if too large
        if max(image.size) > 1024:
            ratio = 1024.0 / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
            original_image = original_image.resize(new_size, Image.LANCZOS)

        results = {}

        # Process with deep learning model if specified
        if model_name and model_name in models:
            # Check if we're using a color model
            is_color_model = model_name.startswith('color_')

            # Convert to tensor and normalize
            transform = ToTensor()

            # Always convert to the correct format based on model type
            if is_color_model:
                # For color models, use RGB image
                if image.mode != 'RGB':
                    image_rgb = image.convert('RGB')
                else:
                    image_rgb = image

                image_tensor = transform(image_rgb).unsqueeze(0).to(device)
            else:
                # For grayscale models, always use grayscale
                if image.mode != 'L':
                    image_gray = image.convert('L')
                else:
                    image_gray = image

                image_tensor = transform(image_gray).unsqueeze(0).to(device)

            # Prepare for processing
            with torch.no_grad():
                output = models[model_name](image_tensor)
                output = torch.clamp(output, 0, 1)

            # Convert back to PIL Image
            dl_result = ToPILImage()(output.squeeze(0).cpu())

            # If we need to preserve color but used a grayscale model
            if preserve_color and not is_color_model and image.mode == 'RGB':
                # Use the grayscale result as luminance and preserve original color
                # Convert to YCbCr color space
                original_ycbcr = image.convert('YCbCr')

                # Get the super-resolved Y channel
                result_y = dl_result.convert('L')

                # Get the original Cb and Cr channels
                y, cb, cr = original_ycbcr.split()

                # Resize Cb and Cr to match the new Y dimensions if needed
                if result_y.size != cb.size:
                    cb = cb.resize(result_y.size, Image.BICUBIC)
                    cr = cr.resize(result_y.size, Image.BICUBIC)

                # Merge channels and convert back to RGB
                result_ycbcr = Image.merge('YCbCr', [result_y, cb, cr])
                dl_result = result_ycbcr.convert('RGB')

            results['deep_learning'] = dl_result

        # Process with classical method if specified
        if classical_method and classical_method in classical_methods:
            try:
                # For classical methods, we need to handle color preservation separately
                if preserve_color and image.mode == 'RGB':
                    # Convert to YCbCr and process only Y channel
                    ycbcr = image.convert('YCbCr')
                    y, cb, cr = ycbcr.split()

                    # Process Y channel
                    y_processed = classical_methods[classical_method].process(y)
                    if isinstance(y_processed, np.ndarray):
                        y_processed = Image.fromarray(y_processed.astype(np.uint8))

                    # Resize Cb and Cr to match the new Y dimensions
                    new_size = y_processed.size
                    cb = cb.resize(new_size, Image.BICUBIC)
                    cr = cr.resize(new_size, Image.BICUBIC)

                    # Merge channels and convert back to RGB
                    cl_result = Image.merge('YCbCr', [y_processed, cb, cr]).convert('RGB')
                else:
                    # Standard processing (grayscale)
                    if image.mode != 'L':
                        image_gray = image.convert('L')
                    else:
                        image_gray = image

                    cl_result = classical_methods[classical_method].process(image_gray)
                    if isinstance(cl_result, np.ndarray):
                        if cl_result.dtype == np.float64 or cl_result.dtype == np.float32:
                            # Normalize float arrays to 0-255 range
                            cl_result = ((cl_result - cl_result.min()) * 255 / (cl_result.max() - cl_result.min())).astype(np.uint8)
                        cl_result = Image.fromarray(cl_result)
            except Exception as e:
                st.error(f"Error in classical method processing: {e}")
                # Create a blank image as fallback
                cl_result = Image.new('L', image.size, 128)
                if image.mode == 'RGB':
                    cl_result = cl_result.convert('RGB')

            results['classical'] = cl_result

        return results
    except Exception as e:
        st.error(f"Error in process_image: {e}")
        return None

def set_language(lang):
    """Set the application language"""
    if lang in ['en', 'zh']:
        st.session_state.language = lang
        st.experimental_rerun()

def display_header():
    """Display the application header with language selection"""
    lang = st.session_state.language

    col1, col2 = st.columns([6, 1])

    with col1:
        st.title(get_text('app_title', lang))
        st.markdown(get_text('tagline', lang))

    with col2:
        if st.button("English" if lang == 'zh' else "中文"):
            set_language('en' if lang == 'zh' else 'zh')

def single_image_processing():
    """Handle single image processing UI and logic"""
    lang = st.session_state.language

    st.header(get_text('single_processing', lang))

    # Check if models are available
    if not models:
        st.warning("No models are loaded. Please wait for models to load or check for errors.")
        return

    # Upload image
    uploaded_file = st.file_uploader(
        get_text('upload_image', lang),
        type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        help=get_text('drag_drop', lang)
    )

    # Processing options
    st.subheader(get_text('processing_options', lang))

    col1, col2 = st.columns(2)

    with col1:
        model_options = list(models.keys())
        if not model_options:
            st.error("No models available. Please check the logs for errors.")
            model_name = None
        else:
            model_name = st.selectbox(
                get_text('dl_model', lang),
                options=model_options,
                format_func=lambda x: get_text(f'model_{x}', lang)
            )

    with col2:
        classical_options = list(classical_methods.keys())
        classical_method = st.selectbox(
            get_text('classical_method', lang),
            options=['none'] + classical_options,
            format_func=lambda x: x.capitalize() if x != 'none' else 'None'
        )
        if classical_method == 'none':
            classical_method = None

    preserve_color = st.checkbox(
        get_text('preserve_color', lang),
        help=get_text('preserve_color_desc', lang)
    )

    # Process button
    process_button = st.button(get_text('process_image', lang), type="primary")

    # Handle processing
    if uploaded_file is not None and process_button:
        with st.spinner(get_text('processing', lang)):
            # Read image
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))

            # Save original image
            timestamp = int(time.time())
            filename = f"{timestamp}_{uploaded_file.name}"
            original_path = os.path.join(UPLOAD_DIR, filename)
            image.save(original_path)

            # Process image
            results = process_image(image, model_name, classical_method, preserve_color)

            if results:
                # Display results
                st.subheader(get_text('result', lang))

                # Save results
                result_images = {}
                metrics = {}

                if 'deep_learning' in results:
                    dl_path = os.path.join(RESULTS_DIR, f"{timestamp}_{model_name}_{uploaded_file.name}")
                    results['deep_learning'].save(dl_path)
                    result_images['deep_learning'] = {
                        'path': dl_path,
                        'image': results['deep_learning']
                    }

                    # Calculate metrics
                    try:
                        dl_metrics = evaluate_image_quality(image, results['deep_learning'])
                        metrics['deep_learning'] = dl_metrics
                    except Exception as e:
                        st.error(f"Error calculating metrics: {e}")
                        metrics['deep_learning'] = {
                            'psnr': 'N/A',
                            'ssim': 'N/A',
                            'quality': 'error'
                        }

                if 'classical' in results:
                    cl_path = os.path.join(RESULTS_DIR, f"{timestamp}_{classical_method}_{uploaded_file.name}")
                    results['classical'].save(cl_path)
                    result_images['classical'] = {
                        'path': cl_path,
                        'image': results['classical']
                    }

                    # Calculate metrics
                    try:
                        cl_metrics = evaluate_image_quality(image, results['classical'])
                        metrics['classical'] = cl_metrics
                    except Exception as e:
                        st.error(f"Error calculating metrics: {e}")
                        metrics['classical'] = {
                            'psnr': 'N/A',
                            'ssim': 'N/A',
                            'quality': 'error'
                        }

                # Store in session state
                st.session_state.processed_images.append({
                    'original': {
                        'path': original_path,
                        'image': image
                    },
                    'results': result_images,
                    'metrics': metrics,
                    'timestamp': timestamp,
                    'filename': uploaded_file.name
                })

                # Display the most recent result
                display_results(st.session_state.processed_images[-1], lang)
            else:
                st.error(get_text('error_processing', lang))

def display_results(result_data, lang):
    """Display processing results with metrics"""
    # Display original vs processed images
    st.subheader(get_text('result', lang))

    # Create tabs for different results
    if 'deep_learning' in result_data['results'] and 'classical' in result_data['results']:
        tab1, tab2 = st.tabs(["Deep Learning", "Classical Method"])

        with tab1:
            display_single_result(
                result_data['original']['image'],
                result_data['results']['deep_learning']['image'],
                result_data['metrics']['deep_learning'] if 'deep_learning' in result_data['metrics'] else None,
                lang
            )

        with tab2:
            display_single_result(
                result_data['original']['image'],
                result_data['results']['classical']['image'],
                result_data['metrics']['classical'] if 'classical' in result_data['metrics'] else None,
                lang
            )
    elif 'deep_learning' in result_data['results']:
        display_single_result(
            result_data['original']['image'],
            result_data['results']['deep_learning']['image'],
            result_data['metrics']['deep_learning'] if 'deep_learning' in result_data['metrics'] else None,
            lang
        )
    elif 'classical' in result_data['results']:
        display_single_result(
            result_data['original']['image'],
            result_data['results']['classical']['image'],
            result_data['metrics']['classical'] if 'classical' in result_data['metrics'] else None,
            lang
        )

def display_single_result(original_image, processed_image, metrics, lang):
    """Display a single result with metrics"""
    # Use image comparison widget if available, otherwise use columns
    if HAS_IMAGE_COMPARISON:
        try:
            image_comparison(
                img1=original_image,
                img2=processed_image,
                label1=get_text('original_image', lang),
                label2=get_text('result', lang),
                width=700
            )
        except Exception as e:
            logger.error(f"Error displaying image comparison: {e}")
            st.error("Error displaying image comparison. Falling back to standard display.")
            # Fallback to columns
            display_images_in_columns(original_image, processed_image, lang)
    else:
        # Fallback to columns
        display_images_in_columns(original_image, processed_image, lang)

    # Display metrics if available
    if metrics:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=get_text('psnr_title', lang),
                value=f"{round(metrics['psnr'], 2)} dB" if isinstance(metrics['psnr'], (int, float)) else metrics['psnr']
            )

        with col2:
            st.metric(
                label=get_text('ssim_title', lang),
                value=f"{round(metrics['ssim'], 4)}" if isinstance(metrics['ssim'], (int, float)) else metrics['ssim']
            )

        with col3:
            st.metric(
                label="Quality",
                value=metrics['quality'].capitalize() if 'quality' in metrics else 'N/A'
            )

def display_images_in_columns(original_image, processed_image, lang):
    """Display two images side by side in columns"""
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption=get_text('original_image', lang), use_column_width=True)
    with col2:
        st.image(processed_image, caption=get_text('result', lang), use_column_width=True)

def batch_processing():
    """Handle batch image processing UI and logic"""
    lang = st.session_state.language

    st.header(get_text('batch_processing', lang))

    # Check if models are available
    if not models:
        st.warning("No models are loaded. Please wait for models to load or check for errors.")
        return

    # Upload multiple images
    uploaded_files = st.file_uploader(
        get_text('upload_multiple', lang),
        type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'],
        accept_multiple_files=True,
        help=get_text('drag_drop_multiple', lang)
    )

    # Display selected files
    if uploaded_files:
        st.text(f"{len(uploaded_files)} files selected")
    else:
        st.text(get_text('no_files', lang))

    # Processing options
    st.subheader(get_text('batch_options', lang))

    col1, col2 = st.columns(2)

    with col1:
        model_options = list(models.keys())
        if not model_options:
            st.error("No models available. Please check the logs for errors.")
            model_name = None
        else:
            model_name = st.selectbox(
                get_text('dl_model', lang),
                options=model_options,
                format_func=lambda x: get_text(f'model_{x}', lang),
                key="batch_model"
            )

    with col2:
        preserve_color = st.checkbox(
            get_text('preserve_color', lang),
            help=get_text('preserve_color_desc', lang),
            key="batch_preserve_color"
        )

    # Process button
    process_batch_button = st.button(get_text('process_batch', lang), type="primary")

    # Handle batch processing
    if uploaded_files and process_batch_button:
        with st.spinner(get_text('processing', lang)):
            # Update batch processor with selected model
            if model_name in models:
                global batch_processor
                batch_processor = BatchProcessor(
                    model=models[model_name],
                    device=device,
                    batch_size=4,
                    scale_factor=2,
                    save_dir=RESULTS_DIR,
                    preserve_color=preserve_color
                )

            # Create batch directory
            timestamp = int(time.time())
            batch_dir = os.path.join(UPLOAD_DIR, f"batch_{timestamp}")
            os.makedirs(batch_dir, exist_ok=True)

            # Save uploaded files
            processed_files = []
            for file in uploaded_files:
                file_path = os.path.join(batch_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                processed_files.append(file_path)

            # Process batch
            results = batch_processor.process_directory(batch_dir)

            # Calculate average metrics
            avg_metrics = {'psnr': 0, 'ssim': 0, 'count': 0, 'quality_counts': {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0, 'error': 0}}

            for file_path in processed_files:
                try:
                    original_img = Image.open(file_path)
                    filename = os.path.basename(file_path)

                    # Find corresponding result
                    result_item = next((r for r in results if r['filename'] == filename), None)
                    if result_item:
                        result_path = result_item['path']
                        result_img = Image.open(result_path)

                        # Calculate metrics
                        metrics = evaluate_image_quality(original_img, result_img)
                        avg_metrics['psnr'] += metrics['psnr']
                        avg_metrics['ssim'] += metrics['ssim']
                        avg_metrics['quality_counts'][metrics['quality']] += 1
                        avg_metrics['count'] += 1
                except Exception as e:
                    st.error(f"Error calculating metrics for {file_path}: {e}")

            # Calculate averages
            if avg_metrics['count'] > 0:
                avg_metrics['psnr'] = round(avg_metrics['psnr'] / avg_metrics['count'], 2)
                avg_metrics['ssim'] = round(avg_metrics['ssim'] / avg_metrics['count'], 4)

                # Determine overall quality based on the most common quality level
                quality_counts = avg_metrics['quality_counts']
                max_quality = max(quality_counts.items(), key=lambda x: x[1])
                avg_metrics['quality'] = max_quality[0]
            else:
                avg_metrics = None

            # Store results in session state
            st.session_state.batch_results = {
                'results': results,
                'avg_metrics': avg_metrics,
                'timestamp': timestamp,
                'count': len(results)
            }

            # Display batch results
            display_batch_results(st.session_state.batch_results, lang)

def display_batch_results(batch_data, lang):
    """Display batch processing results"""
    st.subheader(get_text('processed_images', lang).format(batch_data['count']))

    # Display average metrics
    if batch_data['avg_metrics']:
        st.subheader(get_text('avg_metrics', lang))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label=get_text('psnr_title', lang),
                value=f"{batch_data['avg_metrics']['psnr']} dB"
            )

        with col2:
            st.metric(
                label=get_text('ssim_title', lang),
                value=f"{batch_data['avg_metrics']['ssim']}"
            )

        with col3:
            st.metric(
                label="Quality",
                value=batch_data['avg_metrics']['quality'].capitalize()
            )

    # Display gallery of results
    st.subheader("Results Gallery")

    # Create a grid of images
    cols = 3
    rows = (len(batch_data['results']) + cols - 1) // cols

    for i in range(rows):
        row_cols = st.columns(cols)
        for j in range(cols):
            idx = i * cols + j
            if idx < len(batch_data['results']):
                result = batch_data['results'][idx]
                with row_cols[j]:
                    st.image(result['path'], caption=result['filename'], use_column_width=True)
                    st.download_button(
                        label=get_text('download', lang),
                        data=open(result['path'], "rb").read(),
                        file_name=result['filename'],
                        mime="image/png"
                    )

def main():
    """Main application function"""
    try:
        # Display header
        display_header()

        # Add a sidebar with information
        with st.sidebar:
            st.title("About")
            st.markdown("This application uses deep learning to enhance the resolution of medical images.")
            st.markdown("---")
            st.markdown("### System Information")

            # Show deployment environment
            if IS_CLOUD_DEPLOYMENT:
                st.warning("⚠️ Cloud Deployment Mode")
                st.info("Limited models available to conserve memory")
            else:
                st.success("🖥️ Local Deployment Mode")
                st.info("All models available")

            st.info(f"Device: {device}")
            st.info(f"PyTorch version: {torch.__version__}")

            # Add a button to clear cache
            if st.button("Clear Cache"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    if key != 'language':  # Preserve language setting
                        del st.session_state[key]
                st.success("Cache cleared!")
                st.experimental_rerun()

        # Load models if not already loaded
        if not models:
            with st.spinner("Loading models..."):
                try:
                    load_models()
                    if not models:  # If no models were loaded
                        st.warning("No models were loaded. Some functionality may be limited.")
                except Exception as e:
                    st.error(f"Error loading models: {e}")
                    st.warning("Continuing with limited functionality.")

        # Create tabs for single and batch processing
        tab1, tab2, tab3 = st.tabs([
            get_text('single_processing', st.session_state.language),
            get_text('batch_processing', st.session_state.language),
            get_text('technology', st.session_state.language)
        ])

        with tab1:
            try:
                single_image_processing()
            except Exception as e:
                st.error(f"Error in single image processing: {e}")
                st.info("Please try again or check the logs for more information.")

        with tab2:
            try:
                batch_processing()
            except Exception as e:
                st.error(f"Error in batch processing: {e}")
                st.info("Please try again or check the logs for more information.")

        with tab3:
            try:
                display_technology_info()
            except Exception as e:
                st.error(f"Error displaying technology information: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please refresh the page and try again.")

def display_technology_info():
    """Display information about the technology"""
    lang = st.session_state.language

    st.header(get_text('tech_title', lang))
    st.markdown(get_text('tech_subtitle', lang))

    # Key features
    st.subheader(get_text('key_features', lang))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"**{get_text('deep_learning', lang)}**")
        st.markdown(get_text('deep_learning_desc', lang))

    with col2:
        st.markdown(f"**{get_text('high_performance', lang)}**")
        st.markdown(get_text('high_performance_desc', lang))

    with col3:
        st.markdown(f"**{get_text('accessibility', lang)}**")
        st.markdown(get_text('accessibility_desc', lang))

    # Technology overview
    st.subheader(get_text('tech_overview', lang))
    st.markdown(get_text('tech_overview_desc', lang))

    # Models
    st.subheader(get_text('models_title', lang))
    st.markdown(get_text('models_desc', lang))

    # SRCNN
    st.markdown(f"**{get_text('srcnn_title', lang)}**")
    st.markdown(get_text('srcnn_desc', lang))
    st.markdown(f"- {get_text('srcnn_feature_extraction', lang)}")
    st.markdown(f"- {get_text('srcnn_nonlinear_mapping', lang)}")
    st.markdown(f"- {get_text('srcnn_reconstruction', lang)}")

    # ESPCN
    st.markdown(f"**{get_text('espcn_title', lang)}**")
    st.markdown(get_text('espcn_desc', lang))

    # EDSR
    st.markdown(f"**{get_text('edsr_title', lang)}**")
    st.markdown(get_text('edsr_desc', lang))
    st.markdown(get_text('edsr_effective', lang))

    # Image quality metrics
    st.subheader(get_text('image_quality_metrics', lang))

    # PSNR
    st.markdown(f"**{get_text('psnr_metric', lang)}**")
    st.markdown(get_text('psnr_metric_desc', lang))
    st.markdown(get_text('psnr_formula_desc', lang))
    st.markdown(get_text('psnr_typical_values', lang))

    # SSIM
    st.markdown(f"**{get_text('ssim_metric', lang)}**")
    st.markdown(get_text('ssim_metric_desc', lang))
    st.markdown(get_text('ssim_perception', lang))
    st.markdown(get_text('ssim_typical_values', lang))

    st.markdown(get_text('metrics_auto_calc', lang))

if __name__ == "__main__":
    main()
