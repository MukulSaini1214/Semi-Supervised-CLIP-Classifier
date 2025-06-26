# Semi-Supervised-CLIP-Classifier

A semi-supervised continual learning image classifier that leverages **CLIP** for feature extraction, **PCA** for dimensionality reduction, and **Logistic Regression** for classification. The model is trained sequentially across multiple datasets (D1–D20) and utilizes pseudo-labeling to learn from unlabeled data.

## Project Overview

- **Feature Extractor**: OpenAI CLIP
- **Dimensionality Reduction**: PCA (e.g., 256 components)
- **Classifier**: Logistic Regression (Scikit-learn)
- **Learning Strategy**: Continual + Semi-Supervised Learning
- **Storage**: All features and models are cached for reuse

## Key Features

- Uses **CLIP embeddings** for powerful zero-shot visual representation
- Reduces feature dimensionality with **PCA**
- Applies **Logistic Regression** on reduced features
- Supports **semi-supervised learning** using pseudo-labels
- Trains incrementally across datasets (e.g., D1–D10, D11–D20)
- Efficient **caching system** to avoid recomputation of embeddings


## Workflow

1. **Extract features** from images using CLIP
2. **Reduce** features using PCA (fit on D1 or early datasets)
3. **Train classifier** on labeled features
4. For unlabeled data:
   - Predict with current classifier
   - Select high-confidence pseudo-labels (e.g., confidence > 0.8)
   - Retrain classifier with pseudo-labeled data
5. **Evaluate** performance after each dataset increment

