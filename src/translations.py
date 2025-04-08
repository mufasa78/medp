"""
Translation dictionary for the Medical Image Super-Resolution application.
Contains translations for English and Chinese.
"""

translations = {
    'en': {
        # Navigation
        'app_title': 'Medical Image Super-Resolution',
        'home': 'Home',
        'technology': 'Technology',
        'about': 'About',

        # Home page
        'tagline': 'Enhance the resolution of medical images using advanced deep learning techniques.',
        'description': 'Upload your medical images and see the enhanced results in real-time. Compare different super-resolution methods and process images in batch.',
        'single_processing': 'Single Image',
        'batch_processing': 'Batch Processing',

        # Model descriptions
        'model_srcnn': 'SRCNN: Fast and lightweight model for basic super-resolution.',
        'model_espcn': 'ESPCN: Efficient model with pixel shuffling for real-time applications.',
        'model_edsr': 'EDSR: Enhanced Deep SR with residual blocks for better detail preservation.',
        'model_rcan': 'RCAN: Advanced model with channel attention for capturing fine details.',
        'model_srresnet': 'SRResNet: Residual network architecture with skip connections.',
        'model_color_srcnn': 'Color SRCNN: SRCNN model adapted for color image processing.',
        'model_color_espcn': 'Color ESPCN: ESPCN model adapted for color image processing.',
        'model_color_edsr': 'Color EDSR: EDSR model adapted for color image processing.',
        'model_color_rcan': 'Color RCAN: RCAN model adapted for color image processing.',
        'model_color_srresnet': 'Color SRResNet: SRResNet model adapted for color image processing.',

        # Upload section
        'upload_image': 'Upload Image',
        'drag_drop': 'Drag & drop your image here or click to browse',
        'processing_options': 'Processing Options',
        'dl_model': 'Deep Learning Model',
        'classical_method': 'Compare with Classical Method',
        'preserve_color': 'Preserve original colors',
        'preserve_color_desc': 'When enabled, the original color information will be preserved in the super-resolved image.',
        'process_image': 'Process Image',
        'processing': 'Processing your image...',

        # Batch processing
        'upload_multiple': 'Upload Multiple Images',
        'drag_drop_multiple': 'Drag & drop multiple images here or click to browse',
        'no_files': 'No files selected',
        'batch_options': 'Batch Processing Options',
        'process_batch': 'Process Batch',

        # Results
        'original_image': 'Original Image',
        'reference_image': 'Reference image for quality metrics',
        'result': 'Result',
        'download': 'Download',
        'processed_images': 'Processed {0} Images',
        'avg_metrics': 'Average Quality Metrics:',

        # Metrics
        'psnr': 'PSNR: {0} dB',
        'psnr_title': 'Peak Signal-to-Noise Ratio (higher is better)',
        'ssim': 'SSIM: {0}',
        'ssim_title': 'Structural Similarity Index (higher is better)',

        # Technology page
        'tech_title': 'Our Technology',
        'tech_subtitle': 'Explore the advanced AI technology behind our medical image super-resolution system',
        'key_features': 'Key Features',
        'deep_learning': 'Deep Learning',
        'deep_learning_desc': 'Utilizes advanced neural networks trained on medical imaging datasets',
        'high_performance': 'High Performance',
        'high_performance_desc': 'Optimized for speed and accuracy with GPU acceleration',
        'accessibility': 'Accessibility',
        'accessibility_desc': 'Web-based interface makes the technology available anywhere',
        'tech_overview': 'Technology Overview',
        'tech_overview_desc': 'Our medical image super-resolution system uses state-of-the-art deep learning techniques to enhance the quality and resolution of medical images. The technology is designed specifically for medical imaging, with a focus on preserving diagnostic features while improving clarity and detail.',
        'image_quality_metrics': 'Image Quality Metrics',
        'models_title': 'Super-Resolution Models',
        'models_desc': 'We implement several state-of-the-art deep learning models for super-resolution:',
        'classical_methods_title': 'Classical Methods',
        'classical_methods_desc': 'In addition to deep learning approaches, we also implement traditional image processing techniques:',
        'training_process': 'Training Process',
        'training_process_desc': 'Our models are trained using a specialized pipeline designed for medical images:',

        # About page
        'about_title': 'About Our Project',
        'about_subtitle': 'Learn about our mission, team, and the story behind our medical image enhancement technology',
        'mission': 'Our Mission',
        'mission_desc': 'Our mission is to improve medical imaging accessibility and quality worldwide through advanced AI technology.',
        'mission_points': 'We aim to:',
        'mission_point1': 'Enhance diagnostic capabilities through improved image resolution',
        'mission_point2': 'Make advanced image processing accessible to medical facilities worldwide',
        'mission_point3': 'Contribute to better patient outcomes through improved imaging technology',
        'mission_belief': 'We believe that by applying cutting-edge AI and deep learning techniques to medical imaging, we can make a significant impact on healthcare quality and accessibility.',
        'project_history': 'Project History',
        'inception': 'Project Inception',
        'inception_desc': 'The project began as a research initiative to explore the application of super-resolution techniques to medical imaging. Initial research focused on evaluating different neural network architectures for image enhancement.',
        'prototype': 'First Prototype',
        'prototype_desc': 'Development of the first SRCNN prototype for medical image enhancement. Early tests showed promising results with X-ray and MRI images.',
        'expansion': 'Model Expansion',
        'expansion_desc': 'Added additional models (EDSR, RCAN) and implemented comparative analysis tools. Began collecting feedback from medical professionals.',
        'web_interface': 'Web Interface',
        'web_interface_desc': 'Developed the web application to make the technology accessible to medical professionals without requiring technical expertise.',
        'testimonials': 'Testimonials',
        'team': 'Our Team',
        'lead_developer': 'Lead Developer',
        'lead_developer_desc': 'Responsible for the core super-resolution algorithms and model architecture design.',
        'ml_engineer': 'ML Engineer',
        'ml_engineer_desc': 'Specializes in training and optimizing deep learning models for medical imaging applications.',
        'medical_advisor': 'Medical Advisor',
        'medical_advisor_desc': 'Provides clinical expertise to ensure the technology meets real-world medical needs.',

        # Technology page specific
        'srcnn_title': 'Super-Resolution Convolutional Neural Network (SRCNN)',
        'srcnn_desc': 'SRCNN is our primary model architecture, designed to efficiently enhance image resolution while preserving important medical details. The network consists of three main components:',
        'srcnn_feature_extraction': 'Feature Extraction: The first layer extracts patches from the low-resolution input and represents them as feature maps.',
        'srcnn_nonlinear_mapping': 'Non-linear Mapping: The middle layer maps these feature representations to high-resolution patch representations.',
        'srcnn_reconstruction': 'Reconstruction: The final layer aggregates the predictions to produce the high-resolution output.',
        'srcnn_diagram': 'SRCNN Architecture Diagram',

        'espcn_title': 'Efficient Sub-Pixel Convolutional Neural Network (ESPCN)',
        'espcn_desc': 'ESPCN is an alternative architecture that uses sub-pixel convolution for efficient upscaling. This model processes the image in the low-resolution space and only upscales at the very end, making it computationally efficient.',
        'espcn_diagram': 'ESPCN Architecture Diagram',

        'edsr_title': 'Enhanced Deep Super-Resolution Network (EDSR)',
        'edsr_desc': 'EDSR is our most advanced model, designed for high-quality super-resolution. It removes unnecessary modules from conventional residual networks and expands the model size while stabilizing the training process.',
        'edsr_diagram': 'EDSR Architecture Diagram',
        'edsr_effective': 'This model is particularly effective for complex medical images where fine details are critical for diagnosis.',

        'psnr_metric': 'PSNR',
        'psnr_metric_desc': 'Peak Signal-to-Noise Ratio measures the ratio between the maximum possible power of a signal and the power of corrupting noise',
        'psnr_formula_desc': 'Where MAX₁ is the maximum pixel value (255 for 8-bit images) and MSE is the Mean Squared Error between the original and processed images.',
        'psnr_typical_values': 'Typical values for medical images range from 25-45 dB, with higher values indicating better quality.',

        'ssim_metric': 'SSIM',
        'ssim_metric_desc': 'Structural Similarity Index measures the similarity between two images based on structural information, luminance, and contrast. Values range from -1 to 1, with 1 indicating perfect similarity.',
        'ssim_perception': 'Unlike PSNR, SSIM considers human visual perception, making it more aligned with perceived image quality.',
        'ssim_typical_values': 'For medical images, SSIM values above 0.85 typically indicate good quality preservation.',

        'metrics_auto_calc': 'Our application automatically calculates and displays these metrics for each processed image, allowing you to objectively compare different super-resolution methods.',

        'dataset_title': 'Dataset',
        'dataset_desc': 'The models are trained on a diverse dataset of medical images, including:',
        'dataset_xray': 'X-rays (chest, bone, dental)',
        'dataset_mri': 'MRI scans (brain, spine, joints)',
        'dataset_ct': 'CT scans (various body regions)',
        'dataset_ultrasound': 'Ultrasound images',

        'training_methodology': 'Training Methodology',
        'training_methodology_desc': 'We employ several advanced techniques to optimize model performance:',
        'mixed_precision': 'Mixed Precision Training: Accelerates training while maintaining accuracy',
        'learning_rate': 'Learning Rate Scheduling: Adaptive learning rates for optimal convergence',
        'early_stopping': 'Early Stopping: Prevents overfitting by monitoring validation performance',
        'gradient_clipping': 'Gradient Clipping: Stabilizes training by preventing exploding gradients',
        'data_augmentation': 'Data Augmentation: Enhances model generalization through synthetic variations',

        'clinical_applications': 'Clinical Applications',
        'clinical_applications_desc': 'Our super-resolution technology has several important applications in clinical settings:',
        'legacy_enhancement': 'Legacy Equipment Enhancement',
        'legacy_enhancement_desc': 'Improves the quality of images from older medical imaging equipment, extending their useful life and improving diagnostic capabilities without hardware upgrades.',
        'detail_enhancement': 'Detail Enhancement',
        'detail_enhancement_desc': 'Enhances fine details in medical images that may be critical for accurate diagnosis, such as small lesions, fracture lines, or tissue abnormalities.',
        'archive_restoration': 'Archive Restoration',
        'archive_restoration_desc': 'Enhances the quality of archived medical images that may have degraded over time or were originally captured at lower resolutions.',
        'telemedicine': 'Telemedicine Support',
        'telemedicine_desc': 'Improves the quality of transmitted images in telemedicine applications, ensuring remote specialists can make accurate assessments.',
        'xray_enhancement': 'Example: X-ray Enhancement',
        'original': 'Original',
        'enhanced': 'Enhanced',

        # About page specific
        'ongoing_development': 'Ongoing Development',
        'ongoing_development_desc': 'Continuing to improve model performance, expand supported image types, and enhance the user interface based on user feedback.',
        'january_2023': 'January 2023',
        'march_2023': 'March 2023',
        'june_2023': 'June 2023',
        'october_2023': 'October 2023',
        'january_2024': 'January 2024',
        'present': 'Present',

        'testimonial1': '"This technology has significantly improved our ability to analyze subtle details in older X-ray images. It\'s like getting a hardware upgrade without changing our equipment."',
        'testimonial1_author': '— Dr. Sarah Johnson, Radiologist',
        'testimonial2': '"We\'ve been using this tool for enhancing ultrasound images, and the results are remarkable. The enhanced clarity helps us make more confident diagnoses, especially in challenging cases."',
        'testimonial2_author': '— Dr. Michael Chen, Sonographer',
        'testimonial3': '"As a rural healthcare provider with limited resources, this technology has been invaluable. It helps us get more diagnostic value from our existing imaging equipment."',
        'testimonial3_author': '— Dr. Robert Patel, Rural Health Clinic Director',

        # Footer
        'footer_copyright': '© 2025 Medical Super-Resolution Project',
        'footer_powered': 'Powered by PyTorch & Flask',
        'medical_advisor_desc': 'Provides clinical expertise and ensures the technology meets real-world medical needs.'
    },

    'zh': {
        # Navigation
        'app_title': '医学图像超分辨率',
        'home': '首页',
        'technology': '技术',
        'about': '关于我们',

        # Home page
        'tagline': '使用先进的深度学习技术提高医学图像的分辨率。',
        'description': '上传您的医学图像，实时查看增强效果。比较不同的超分辨率方法，批量处理图像。',
        'single_processing': '单张图像',
        'batch_processing': '批量处理',

        # Model descriptions
        'model_srcnn': 'SRCNN: 用于基本超分辨率的快速轻量级模型。',
        'model_espcn': 'ESPCN: 使用像素混洗的高效模型，适用于实时应用。',
        'model_edsr': 'EDSR: 增强型深度超分辨率网络，具有残差块，可更好地保留细节。',
        'model_rcan': 'RCAN: 具有通道注意力机制的高级模型，可捕获精细细节。',
        'model_srresnet': 'SRResNet: 具有跳跃连接的残差网络架构。',
        'model_color_srcnn': '彩色SRCNN: 适用于彩色图像处理的SRCNN模型。',
        'model_color_espcn': '彩色ESPCN: 适用于彩色图像处理的ESPCN模型。',
        'model_color_edsr': '彩色EDSR: 适用于彩色图像处理的EDSR模型。',
        'model_color_rcan': '彩色RCAN: 适用于彩色图像处理的RCAN模型。',
        'model_color_srresnet': '彩色SRResNet: 适用于彩色图像处理的SRResNet模型。',

        # Upload section
        'upload_image': '上传图像',
        'drag_drop': '拖放图像到此处或点击浏览',
        'processing_options': '处理选项',
        'dl_model': '深度学习模型',
        'classical_method': '与传统方法比较',
        'preserve_color': '保留原始颜色',
        'preserve_color_desc': '启用后，超分辨率图像中将保留原始颜色信息。',
        'process_image': '处理图像',
        'processing': '正在处理您的图像...',

        # Batch processing
        'upload_multiple': '上传多张图像',
        'drag_drop_multiple': '拖放多张图像到此处或点击浏览',
        'no_files': '未选择文件',
        'batch_options': '批量处理选项',
        'process_batch': '批量处理',

        # Results
        'original_image': '原始图像',
        'reference_image': '质量指标参考图像',
        'result': '结果',
        'download': '下载',
        'processed_images': '已处理 {0} 张图像',
        'avg_metrics': '平均质量指标:',

        # Metrics
        'psnr': 'PSNR: {0} dB',
        'psnr_title': '峰值信噪比（越高越好）',
        'ssim': 'SSIM: {0}',
        'ssim_title': '结构相似性指数（越高越好）',

        # Technology page
        'tech_title': '我们的技术',
        'tech_subtitle': '探索我们医学图像超分辨率系统背后的先进人工智能技术',
        'key_features': '主要特点',
        'deep_learning': '深度学习',
        'deep_learning_desc': '利用在医学影像数据集上训练的先进神经网络',
        'high_performance': '高性能',
        'high_performance_desc': '通过GPU加速优化速度和准确性',
        'accessibility': '易于访问',
        'accessibility_desc': '基于网络的界面使技术随处可用',
        'tech_overview': '技术概述',
        'tech_overview_desc': '我们的医学图像超分辨率系统使用最先进的深度学习技术来提高医学图像的质量和分辨率。该技术专为医学成像设计，重点是在提高清晰度和细节的同时保留诊断特征。',
        'image_quality_metrics': '图像质量指标',
        'models_title': '超分辨率模型',
        'models_desc': '我们实现了几种最先进的深度学习超分辨率模型：',
        'classical_methods_title': '传统方法',
        'classical_methods_desc': '除了深度学习方法外，我们还实现了传统的图像处理技术：',
        'training_process': '训练过程',
        'training_process_desc': '我们的模型使用专为医学图像设计的专门管道进行训练：',

        # About page
        'about_title': '关于我们的项目',
        'about_subtitle': '了解我们的使命、团队和医学图像增强技术背后的故事',
        'mission': '我们的使命',
        'mission_desc': '我们的使命是通过先进的人工智能技术提高全球医学成像的可及性和质量。',
        'mission_points': '我们的目标是：',
        'mission_point1': '通过提高图像分辨率增强诊断能力',
        'mission_point2': '使先进的图像处理技术对全球医疗机构可及',
        'mission_point3': '通过改进的成像技术促进更好的患者结果',
        'mission_belief': '我们相信，通过将前沿人工智能和深度学习技术应用于医学成像，我们可以对医疗质量和可及性产生重大影响。',
        'project_history': '项目历史',
        'inception': '项目启动',
        'inception_desc': '该项目始于一项研究计划，旨在探索超分辨率技术在医学成像中的应用。初步研究集中在评估不同的神经网络架构以增强图像。',
        'prototype': '首个原型',
        'prototype_desc': '开发了第一个用于医学图像增强的SRCNN原型。早期测试在X射线和MRI图像上显示出有希望的结果。',
        'expansion': '模型扩展',
        'expansion_desc': '添加了额外的模型（EDSR，RCAN）并实现了比较分析工具。开始收集医疗专业人员的反馈。',
        'web_interface': '网络界面',
        'web_interface_desc': '开发了网络应用程序，使医疗专业人员无需技术专长即可使用该技术。',
        'testimonials': '用户评价',
        'team': '我们的团队',
        'lead_developer': '首席开发者',
        'lead_developer_desc': '负责核心超分辨率算法和模型架构设计。',
        'ml_engineer': '机器学习工程师',
        'ml_engineer_desc': '专门训练和优化用于医学成像应用的深度学习模型。',
        'medical_advisor': '医学顾问',
        'medical_advisor_desc': '提供临床专业知识，确保技术满足现实世界的医疗需求。',

        # Technology page specific
        'srcnn_title': '超分辨率卷积神经网络 (SRCNN)',
        'srcnn_desc': 'SRCNN是我们的主要模型架构，旨在有效提高图像分辨率，同时保留重要的医学细节。该网络由三个主要组件组成：',
        'srcnn_feature_extraction': '特征提取：第一层从低分辨率输入中提取图像块并将其表示为特征图。',
        'srcnn_nonlinear_mapping': '非线性映射：中间层将这些特征表示映射到高分辨率图像块表示。',
        'srcnn_reconstruction': '重建：最后一层聚合预测结果，生成高分辨率输出。',
        'srcnn_diagram': 'SRCNN架构图',

        'espcn_title': '高效子像素卷积神经网络 (ESPCN)',
        'espcn_desc': 'ESPCN是一种替代架构，使用子像素卷积进行高效上采样。该模型在低分辨率空间处理图像，仅在最后进行上采样，使其计算效率高。',
        'espcn_diagram': 'ESPCN架构图',

        'edsr_title': '增强型深度超分辨率网络 (EDSR)',
        'edsr_desc': 'EDSR是我们最先进的模型，专为高质量超分辨率设计。它从传统残差网络中移除了不必要的模块，扩展了模型大小，同时稳定了训练过程。',
        'edsr_diagram': 'EDSR架构图',
        'edsr_effective': '这个模型对于细节对诊断至关重要的复杂医学图像特别有效。',

        'psnr_metric': 'PSNR',
        'psnr_metric_desc': '峰值信噪比衡量信号的最大可能功率与干扰噪声功率之间的比率',
        'psnr_formula_desc': '其中MAX₁是最大像素值（8位图像为255），MSE是原始图像和处理后图像之间的均方误差。',
        'psnr_typical_values': '医学图像的典型值范围为25-45 dB，值越高表示质量越好。',

        'ssim_metric': 'SSIM',
        'ssim_metric_desc': '结构相似性指数基于结构信息、亮度和对比度衡量两个图像之间的相似性。值范围从-1到1，1表示完全相似。',
        'ssim_perception': '与PSNR不同，SSIM考虑了人类视觉感知，使其与感知图像质量更加一致。',
        'ssim_typical_values': '对于医学图像，SSIM值高于0.85通常表示良好的质量保存。',

        'metrics_auto_calc': '我们的应用程序自动计算并显示每个处理图像的这些指标，使您能够客观比较不同的超分辨率方法。',

        'dataset_title': '数据集',
        'dataset_desc': '模型在多样化的医学图像数据集上训练，包括：',
        'dataset_xray': 'X射线（胸部、骨骼、牙科）',
        'dataset_mri': 'MRI扫描（大脑、脊椎、关节）',
        'dataset_ct': 'CT扫描（各种身体区域）',
        'dataset_ultrasound': '超声波图像',

        'training_methodology': '训练方法',
        'training_methodology_desc': '我们采用几种先进技术来优化模型性能：',
        'mixed_precision': '混合精度训练：在保持准确性的同时加速训练',
        'learning_rate': '学习率调度：自适应学习率以获得最佳收敛',
        'early_stopping': '早停：通过监控验证性能防止过拟合',
        'gradient_clipping': '梯度裁剪：通过防止梯度爆炸稳定训练',
        'data_augmentation': '数据增强：通过合成变化增强模型泛化能力',

        'clinical_applications': '临床应用',
        'clinical_applications_desc': '我们的超分辨率技术在临床环境中有几个重要应用：',
        'legacy_enhancement': '旧设备增强',
        'legacy_enhancement_desc': '提高旧医学成像设备的图像质量，延长其使用寿命，无需硬件升级即可提高诊断能力。',
        'detail_enhancement': '细节增强',
        'detail_enhancement_desc': '增强医学图像中对准确诊断可能至关重要的精细细节，如小病变、骨折线或组织异常。',
        'archive_restoration': '档案修复',
        'archive_restoration_desc': '提高可能随时间退化或最初以较低分辨率捕获的存档医学图像的质量。',
        'telemedicine': '远程医疗支持',
        'telemedicine_desc': '提高远程医疗应用中传输图像的质量，确保远程专家能够做出准确评估。',
        'xray_enhancement': '示例：X射线增强',
        'original': '原始图像',
        'enhanced': '增强图像',

        # About page specific
        'ongoing_development': '持续开发',
        'ongoing_development_desc': '继续改进模型性能，扩展支持的图像类型，并根据用户反馈增强用户界面。',
        'january_2023': '2023年1月',
        'march_2023': '2023年3月',
        'june_2023': '2023年6月',
        'october_2023': '2023年10月',
        'january_2024': '2024年1月',
        'present': '现在',

        'testimonial1': '"这项技术显著提高了我们分析旧X射线图像中细微细节的能力。这就像在不更换设备的情况下获得了硬件升级。"',
        'testimonial1_author': '— Sarah Johnson医生，放射科医师',
        'testimonial2': '"我们一直在使用这个工具增强超声波图像，结果非常显著。增强的清晰度帮助我们做出更有信心的诊断，尤其是在具有挑战性的案例中。"',
        'testimonial2_author': '— Michael Chen医生，超声检查师',
        'testimonial3': '"作为资源有限的农村医疗提供者，这项技术非常宝贵。它帮助我们从现有的成像设备中获得更多的诊断价值。"',
        'testimonial3_author': '— Robert Patel医生，农村健康诊所主任',

        # Footer
        'footer_copyright': '© 2025 医学超分辨率项目',
        'footer_powered': '由PyTorch和Flask提供支持'
    }
}

def get_text(key, lang='en', *args):
    """
    Get translated text for the given key and language.

    Args:
        key: The translation key
        lang: Language code ('en' or 'zh')
        *args: Format arguments if the text contains placeholders

    Returns:
        Translated text
    """
    if lang not in translations:
        lang = 'en'  # Fallback to English

    text = translations[lang].get(key, translations['en'].get(key, key))

    if args:
        try:
            return text.format(*args)
        except:
            return text

    return text
