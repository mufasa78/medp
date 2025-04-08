/**
 * Client-side translations for the Medical Image Super-Resolution application.
 * This file provides translations for dynamic content that can't be handled by server-side templates.
 */

const translations = {
    'en': {
        // Original image
        'original_image': 'Original Image',
        'reference_image': 'Reference image for quality metrics',

        // Results
        'result': 'Result',
        'download': 'Download',

        // Metrics
        'psnr': 'PSNR: {0} dB',
        'psnr_title': 'Peak Signal-to-Noise Ratio (higher is better)',
        'ssim': 'SSIM: {0}',
        'ssim_title': 'Structural Similarity Index (higher is better)',
        'quality': 'Quality: {0}',
        'quality_title': 'Overall quality assessment based on PSNR and SSIM',
        'quality_excellent': 'Excellent',
        'quality_good': 'Good',
        'quality_acceptable': 'Acceptable',
        'quality_poor': 'Poor',
        'quality_error': 'Error',

        // Batch processing
        'processed_images': 'Processed {0} Images',
        'avg_metrics': 'Average Quality Metrics:',

        // Errors
        'select_image': 'Please select an image file',
        'file_size': '{0} ({1})',

        // Model descriptions
        'model_srcnn': 'SRCNN: Fast and lightweight model for basic super-resolution.',
        'model_espcn': 'ESPCN: Efficient model with pixel shuffling for real-time applications.',
        'model_edsr': 'EDSR: Enhanced Deep SR with residual blocks for better detail preservation.',
        'model_rcan': 'RCAN: Advanced model with channel attention for capturing fine details.',
        'model_srresnet': 'SRResNet: Residual network architecture with skip connections.',
        'model_color_srcnn': 'Color SRCNN: SRCNN model adapted for color image processing.',
        'model_color_espcn': 'Color ESPCN: ESPCN model adapted for color image processing.',
        'model_color_edsr': 'Color EDSR: EDSR model adapted for color image processing.',
        'model_color_rcan': 'Color RCAN: RCAN model adapted for color image processing.',
        'model_color_srresnet': 'Color SRResNet: SRResNet model adapted for color image processing.'
    },

    'zh': {
        // Original image
        'original_image': '原始图像',
        'reference_image': '质量指标参考图像',

        // Results
        'result': '结果',
        'download': '下载',

        // Metrics
        'psnr': 'PSNR: {0} dB',
        'psnr_title': '峰值信噪比（越高越好）',
        'ssim': 'SSIM: {0}',
        'ssim_title': '结构相似性指数（越高越好）',
        'quality': '质量: {0}',
        'quality_title': '基于PSNR和SSIM的整体质量评估',
        'quality_excellent': '优秀',
        'quality_good': '良好',
        'quality_acceptable': '可接受',
        'quality_poor': '较差',
        'quality_error': '错误',

        // Batch processing
        'processed_images': '已处理 {0} 张图像',
        'avg_metrics': '平均质量指标:',

        // Errors
        'select_image': '请选择图像文件',
        'file_size': '{0} ({1})',

        // Model descriptions
        'model_srcnn': 'SRCNN: 用于基本超分辨率的快速轻量级模型。',
        'model_espcn': 'ESPCN: 使用像素混洗的高效模型，适用于实时应用。',
        'model_edsr': 'EDSR: 增强型深度超分辨率网络，具有残差块，可更好地保留细节。',
        'model_rcan': 'RCAN: 具有通道注意力机制的高级模型，可捕获精细细节。',
        'model_srresnet': 'SRResNet: 具有跳跃连接的残差网络架构。',
        'model_color_srcnn': '彩色SRCNN: 适用于彩色图像处理的SRCNN模型。',
        'model_color_espcn': '彩色ESPCN: 适用于彩色图像处理的ESPCN模型。',
        'model_color_edsr': '彩色EDSR: 适用于彩色图像处理的EDSR模型。',
        'model_color_rcan': '彩色RCAN: 适用于彩色图像处理的RCAN模型。',
        'model_color_srresnet': '彩色SRResNet: 适用于彩色图像处理的SRResNet模型。'
    }
};

/**
 * Get translated text for the given key and language.
 *
 * @param {string} key - The translation key
 * @param {string} lang - Language code ('en' or 'zh')
 * @param {...any} args - Format arguments if the text contains placeholders
 * @returns {string} - Translated text
 */
function getClientText(key, lang, ...args) {
    if (!translations[lang]) {
        lang = 'en';  // Fallback to English
    }

    const text = translations[lang][key] || translations['en'][key] || key;

    if (args.length > 0) {
        return text.replace(/\{(\d+)\}/g, (match, index) => {
            const i = parseInt(index);
            return i < args.length ? args[i] : match;
        });
    }

    return text;
}
