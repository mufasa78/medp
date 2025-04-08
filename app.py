import os
import torch
from flask import Flask, request, render_template, jsonify, send_from_directory, session, redirect, url_for
from PIL import Image
import io
import base64
import sys
import time
from werkzeug.utils import secure_filename

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import SRCNN, ESPCN, EDSR, RCAN, SRResNet
from src.color_models import ColorSRCNN, ColorESPCN, ColorEDSR, ColorRCAN, ColorSRResNet
from src.classical_sr import BicubicInterpolation, IBP, NLMeans, EdgeGuidedSR
from src.datasets import BatchProcessor
from src.metrics import evaluate_image_quality
from src.translations import get_text
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
app.secret_key = 'medical_super_resolution_secret_key'  # Required for session

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Model configurations
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
realtime_processor = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Initialize models
def load_models():
    global models, classical_methods, batch_processor

    print(f"Loading models on device: {device}")

    # Load deep learning models
    for model_name, config in MODEL_CONFIGS.items():
        try:
            # Initialize model
            if model_name in ['espcn', 'edsr', 'rcan', 'srresnet']:
                model = config['class'](scale_factor=2).to(device)
            else:
                model = config['class']().to(device)

            # Look for model checkpoint
            if os.path.exists(config['checkpoint']):
                checkpoint = torch.load(config['checkpoint'], map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded {model_name} from {config['checkpoint']}")
            else:
                print(f"No checkpoint found for {model_name}, using untrained model")

            model.eval()
            models[model_name] = model
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    # Initialize classical methods
    for method_name, method_class in CLASSICAL_METHODS.items():
        try:
            classical_methods[method_name] = method_class(scale_factor=2)
            print(f"Initialized classical method: {method_name}")
        except Exception as e:
            print(f"Error initializing {method_name}: {e}")

    # Initialize batch processor with default model (SRCNN if available, otherwise first available)
    default_model = models.get('srcnn', next(iter(models.values())) if models else None)
    if default_model:
        batch_processor = BatchProcessor(
            model=default_model,
            device=device,
            batch_size=4,
            scale_factor=2,
            save_dir=app.config['RESULTS_FOLDER'],
            preserve_color=False
        )
        print("Initialized batch processor")

# Load models at startup
load_models()

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
                print(f"Error in classical method processing: {e}")
                # Create a blank image as fallback
                cl_result = Image.new('L', image.size, 128)
                if image.mode == 'RGB':
                    cl_result = cl_result.convert('RGB')

            results['classical'] = cl_result

        return results
    except Exception as e:
        print(f"Error in process_image: {e}")
        return None

# Helper function to get language from session or request
def get_language():
    # Check if language is set in session
    if 'language' in session:
        return session['language']

    # Check if language is specified in request
    lang = request.args.get('lang')
    if lang in ['en', 'zh']:
        session['language'] = lang
        return lang

    # Default to English
    return 'en'

@app.route('/')
def home():
    lang = get_language()
    return render_template('index.html', lang=lang, get_text=get_text)

@app.route('/about')
def about():
    lang = get_language()
    return render_template('about.html', lang=lang, get_text=get_text)

@app.route('/technology')
def technology():
    lang = get_language()
    return render_template('technology.html', lang=lang, get_text=get_text)

@app.route('/set_language/<lang>')
def set_language(lang):
    if lang in ['en', 'zh']:
        session['language'] = lang
    return redirect(request.referrer or url_for('home'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Get parameters
        model_name = request.form.get('model', 'srcnn')
        classical_method = request.form.get('classical_method', None)
        preserve_color = request.form.get('preserve_color', 'false').lower() == 'true'
        if classical_method == 'none':
            classical_method = None

        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Save original image
        filename = secure_filename(file.filename)
        timestamp = int(time.time())
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        image.save(original_path)

        # Process through model(s)
        results = process_image(image, model_name, classical_method, preserve_color=preserve_color)
        if not results:
            return jsonify({'error': 'Error processing image'}), 500

        response_data = {
            'success': True,
            'original': os.path.relpath(original_path, 'static'),
            'timestamp': timestamp,
            'filename': filename,
            'metrics': {}
        }

        # Save and encode results
        if 'deep_learning' in results:
            dl_path = os.path.join(app.config['RESULTS_FOLDER'], f"{timestamp}_{model_name}_{filename}")
            results['deep_learning'].save(dl_path)
            response_data['deep_learning'] = os.path.relpath(dl_path, 'static')

            # Calculate image quality metrics
            try:
                metrics = evaluate_image_quality(image, results['deep_learning'])
                response_data['metrics']['deep_learning'] = {
                    'psnr': round(metrics['psnr'], 2),
                    'ssim': round(metrics['ssim'], 4),
                    'quality': metrics['quality']
                }
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                response_data['metrics']['deep_learning'] = {
                    'psnr': 'N/A',
                    'ssim': 'N/A',
                    'quality': 'error'
                }

            # Also provide base64 for immediate display
            buffer = io.BytesIO()
            results['deep_learning'].save(buffer, format='PNG')
            buffer.seek(0)
            response_data['deep_learning_base64'] = base64.b64encode(buffer.getvalue()).decode('utf-8')

        if 'classical' in results:
            cl_path = os.path.join(app.config['RESULTS_FOLDER'], f"{timestamp}_{classical_method}_{filename}")
            results['classical'].save(cl_path)
            response_data['classical'] = os.path.relpath(cl_path, 'static')

            # Calculate image quality metrics
            try:
                metrics = evaluate_image_quality(image, results['classical'])
                response_data['metrics']['classical'] = {
                    'psnr': round(metrics['psnr'], 2),
                    'ssim': round(metrics['ssim'], 4),
                    'quality': metrics['quality']
                }
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                response_data['metrics']['classical'] = {
                    'psnr': 'N/A',
                    'ssim': 'N/A',
                    'quality': 'error'
                }

            # Also provide base64 for immediate display
            buffer = io.BytesIO()
            results['classical'].save(buffer, format='PNG')
            buffer.seek(0)
            response_data['classical_base64'] = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify(response_data)
    except Exception as e:
        print(f"Error in upload_file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_process():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected files'}), 400

    try:
        # Get parameters
        model_name = request.form.get('model', 'srcnn')
        preserve_color = request.form.get('preserve_color', 'false').lower() == 'true'

        # Update batch processor with selected model
        if model_name in models:
            global batch_processor
            batch_processor = BatchProcessor(
                model=models[model_name],
                device=device,
                batch_size=4,
                scale_factor=2,
                save_dir=app.config['RESULTS_FOLDER'],
                preserve_color=preserve_color
            )

        # Process each file
        timestamp = int(time.time())
        batch_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)

        processed_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(batch_dir, filename)
                file.save(file_path)
                processed_files.append(file_path)

        # Process the batch
        results = batch_processor.process_directory(batch_dir)

        # Calculate average metrics if possible
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
                print(f"Error calculating metrics for {file_path}: {e}")

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

        return jsonify({
            'success': True,
            'message': f'Processed {len(results)} files',
            'results': results,
            'avg_metrics': avg_metrics
        })
    except Exception as e:
        print(f"Error in batch_process: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models and methods"""
    return jsonify({
        'deep_learning': list(models.keys()),
        'classical': list(classical_methods.keys())
    })

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
