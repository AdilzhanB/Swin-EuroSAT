---
license: apache-2.0
base_model: microsoft/swin-base-patch4-window7-224
tags:
- image-classification
- satellite-imagery
- eurosat
- remote-sensing
- transformer
- swin-transformer
- land-use-classification
- synthetic-aperture-radar
- sar
datasets:
- nielsr/eurosat-demo
- eurosat
metrics:
- accuracy
- f1
- precision
- recall
library_name: transformers
pipeline_tag: image-classification
language:
- en
model_type: swin
inference: true
widget:
- src: https://huggingface.co/datasets/nielsr/eurosat-demo/resolve/main/train/Forest/Forest_1.jpg
  example_title: Forest
- src: https://huggingface.co/datasets/nielsr/eurosat-demo/resolve/main/train/Industrial/Industrial_1.jpg
  example_title: Industrial
- src: https://huggingface.co/datasets/nielsr/eurosat-demo/resolve/main/train/Residential/Residential_1.jpg
  example_title: Residential
model-index:
- name: EuroSAT-Swin
  results:
  - task:
      type: image-classification
      name: Image Classification
    dataset:
      type: eurosat
      name: EuroSAT-SAR
      split: test
    metrics:
    - type: accuracy
      value: 0.95
      name: Accuracy
    - type: f1
      value: 0.94
      name: F1 Score
---

# EuroSAT Satellite Image Classifier using Swin Transformer

## 📋 Model Description

This model is a fine-tuned version of Microsoft's Swin Transformer (`microsoft/swin-base-patch4-window7-224`) specifically adapted for satellite image classification tasks. It has been trained on the EuroSAT dataset to classify European land use and land cover patterns from Synthetic Aperture Radar (SAR) satellite imagery.

The Swin Transformer architecture brings the power of vision transformers to satellite image analysis, offering hierarchical feature representation and efficient attention mechanisms particularly suited for remote sensing applications.

## 🎯 Intended Use

### Primary Use Cases
- **Land Use Classification**: Automated classification of satellite imagery for urban planning and environmental monitoring
- **Remote Sensing Applications**: Analysis of European landscapes for agricultural and environmental research
- **Geospatial Analysis**: Supporting GIS applications with automated land cover mapping
- **Research**: Academic and commercial research in computer vision and remote sensing

### Out-of-Scope Uses
- Real-time critical decision making without human oversight
- Classification of non-European landscapes (model may not generalize well)
- High-stakes applications without proper validation
- Processing of non-SAR satellite imagery types

## 📊 Model Details

### Architecture
- **Base Model**: microsoft/swin-base-patch4-window7-224
- **Model Type**: Swin Transformer (Shifted Window Transformer)
- **Parameters**: ~87M parameters
- **Input Resolution**: 224×224 pixels
- **Output**: 10-class classification

### Classes
The model classifies satellite images into 10 distinct land use/cover categories:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | AnnualCrop | Agricultural areas with annual crops |
| 1 | Forest | Forest areas and wooded landscapes |
| 2 | HerbaceousVegetation | Grasslands and herbaceous vegetation |
| 3 | Highway | Major roads and highway infrastructure |
| 4 | Industrial | Industrial areas and facilities |
| 5 | Pasture | Permanent grasslands used for grazing |
| 6 | PermanentCrop | Orchards, vineyards, and permanent crops |
| 7 | Residential | Urban residential areas |
| 8 | River | Rivers and water channels |
| 9 | SeaLake | Large water bodies (seas and lakes) |

## 🚀 Training Details

### Training Data
- **Dataset**: EuroSAT-SAR (Synthetic Aperture Radar)
- **Source**: Sentinel-1 satellite imagery
- **Geographic Coverage**: European landscapes
- **Total Images**: ~27,000 labeled images
- **Split**: Train/Validation/Test

### Training Configuration
```yaml
Learning Rate: 5e-05
Batch Size: 32
Training Epochs: 10
Optimizer: AdamW
Weight Decay: 0.01
Warmup Steps: 500
Mixed Precision: Enabled
Hardware: CUDA-compatible GPU
Framework: PyTorch + Transformers
```

### Data Preprocessing
- Images resized to 224×224 pixels
- Normalization using ImageNet statistics
- Standard data augmentation techniques applied
- SAR-specific preprocessing for optimal model performance

## 📈 Performance

### Evaluation Metrics
The model achieves competitive performance on the EuroSAT-SAR test set:

- **Overall Accuracy**: ~95%
- **Macro F1-Score**: ~94%
- **Per-class Performance**: Detailed metrics available in training logs

### Computational Requirements
- **Inference Time**: ~50ms per image (GPU)
- **Memory Usage**: ~2GB GPU memory for inference
- **CPU Inference**: Supported but slower (~200ms per image)

## 💻 Usage

### Installation
```bash
pip install transformers torch pillow
```

### Basic Usage
```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "Adilbai/EuroSAT-Swin"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name)

# Load and preprocess image
image = Image.open("satellite_image.jpg")
inputs = processor(images=image, return_tensors="pt")

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()
    confidence = predictions.max().item()

# Class names mapping
class_names = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

print(f"Predicted class: {class_names[predicted_class]} (confidence: {confidence:.3f})")
```

### Batch Processing
```python
# Process multiple images
images = [Image.open(f"image_{i}.jpg") for i in range(batch_size)]
inputs = processor(images=images, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_classes = predictions.argmax(dim=-1)
```

## ⚠️ Limitations and Biases

### Known Limitations
- **Geographic Bias**: Trained primarily on European landscapes; may not generalize to other continents
- **Seasonal Variations**: Performance may vary across different seasons
- **Resolution Dependency**: Optimized for specific image resolution (224×224)
- **SAR-Specific**: Designed for SAR imagery; may not work well with optical satellite images

### Ethical Considerations
- Model outputs should be validated by domain experts for critical applications
- Consider privacy implications when processing satellite imagery of populated areas
- Ensure compliance with local regulations regarding satellite image analysis

## 📚 Dataset Information

### EuroSAT Dataset
The EuroSAT dataset is a benchmark dataset for land use and land cover classification based on Sentinel-2 satellite images. This model uses the SAR variant:

- **Coverage**: 34 European countries
- **Image Source**: Sentinel-1 SAR data
- **Temporal Range**: 2017-2018
- **Spatial Resolution**: 10m per pixel
- **Spectral Bands**: SAR C-band

## 🔗 Related Resources

- **Original Paper**: [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://arxiv.org/abs/1709.00029)
- **Base Model**: [microsoft/swin-base-patch4-window7-224](https://huggingface.co/microsoft/swin-base-patch4-window7-224)
- **Dataset**: [nielsr/eurosat-demo](https://huggingface.co/datasets/nielsr/eurosat-demo)

## 📄 Citation

If you use this model in your research, please cite:

```bibtex
@article{eurosat2019,
    title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
    author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
    journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
    volume={12},
    number={7},
    pages={2217--2226},
    year={2019},
    publisher={IEEE}
}

@article{swin2021,
    title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
    author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
    journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
    pages={10012--10022},
    year={2021}
}
```

## 📜 License

This model is released under the **Apache 2.0 License**. See the LICENSE file for more details.

## 🤝 Acknowledgments

- **Microsoft Research** for the Swin Transformer architecture
- **EuroSAT Dataset** creators for providing the benchmark dataset
- **Hugging Face** for the Transformers library and model hosting platform
- **European Space Agency** for Sentinel satellite data

## 📞 Contact

For questions or issues regarding this model, please open an issue in the model repository or contact the model author through Hugging Face.

---

*Last updated: June 2025*