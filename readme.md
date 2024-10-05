# GPS Coordinate Prediction from Image Features

## Overview

This project focuses on predicting GPS coordinates (latitude and longitude) from image features using machine learning techniques, specifically k-Nearest Neighbors (kNN) and Linear Regression models. The dataset consists of 27.6k geo-tagged images taken in Spain, with image features extracted using OpenAI’s CLIP model. The goal is to analyze and compare the performance of kNN and Linear Regression models in predicting displacement error (in miles) for different dataset sizes and model configurations.

## Dataset

- **Source**: The dataset contains 28,616 geo-tagged images from Mousselly-Sergieh et al. 2014, scraped from Flickr in 2024.
- **Image Features**: Image features were extracted using CLIP ViT-L/14@336px.
- **Training Set**: 27,616 images
- **Test Set**: 1,000 images

## Project Steps

1. **Data Visualization**:
   - Visualized image locations by plotting longitude and latitude values.
   - Applied Principal Component Analysis (PCA) to reduce image features to two dimensions, visualizing image features by longitude.

2. **kNN Regression Implementation**:
   - Implemented kNN regression to predict GPS coordinates (lat, lon) for test images using the nearest neighbors from the training set.
   - Evaluated mean displacement error (miles) across different values of k (nearest neighbors).
   - Conducted grid search to find the best k, both with and without distance weighting.

3. **Baseline Model**:
   - Established a naive baseline by predicting the mean of the training set coordinates, achieving a mean displacement error of 160.3 miles.

4. **Comparison with Linear Regression**:
   - Implemented and compared Linear Regression with kNN on varying training set sizes (ratios from 10% to 100%).
   - For each ratio, compared the mean displacement error of both models.

## Key Results

- **Best k for kNN**: The optimal k value for kNN was found to be **3**.
- **Displacement Error**:
  - **kNN** consistently outperformed Linear Regression as the dataset size increased, achieving a lowest mean displacement error of **131.0 miles** for the full dataset.
  - **Linear Regression** had a lowest mean displacement error of **156.7 miles**.

| **Training Set Ratio** | **Linear Regression Error (miles)** | **kNN Error (miles)** |
|------------------------|------------------------------------|----------------------|
| 0.1 (2,761 samples)     | 184.7                              | 166.8                |
| 0.2 (5,523 samples)     | 166.9                              | 156.6                |
| 0.3 (8,284 samples)     | 161.7                              | 150.5                |
| 0.4 (11,046 samples)    | 159.2                              | 145.5                |
| 0.5 (13,808 samples)    | 158.9                              | 142.6                |
| 0.6 (16,569 samples)    | 158.2                              | 139.4                |
| 0.7 (19,331 samples)    | 157.4                              | 137.7                |
| 0.8 (22,092 samples)    | 157.2                              | 135.9                |
| 0.9 (24,854 samples)    | 156.7                              | 133.7                |
| 1.0 (27,616 samples)    | 156.7                              | 131.0                |

## Files

- **Main Python Script**: `main.py` contains all the implementations for data visualization, kNN regression, and model comparison.
- **Data**: The `im2spain_data.npz` file includes image features and corresponding GPS coordinates for training and testing.
- **Plots**: Various plots are generated to visualize data and model performance:
  - `image_locations.png`: Plot of image locations by GPS coordinates.
  - `pca_features.png`: PCA plot of image features by longitude.
  - `knn_error_weighted.png` and `knn_error_unweighted.png`: Mean displacement error vs. k for weighted and unweighted kNN.
  - `nearest_neighbors.png`: Visualization of the nearest neighbors for a test image.
  - `error_vs_training_size.png`: Comparison of mean displacement error for kNN and Linear Regression as training set size increases.

## How to Run

1. Clone the repository and navigate to the project directory.
2. Ensure you have the required dependencies installed (e.g., NumPy, Matplotlib, Scikit-learn).
3. Download the dataset file `im2spain_data.npz` and place it in the same directory as `main.py`.
4. Run `main.py` to execute all steps, including model evaluation and visualizations.

```bash
python main.py
```

## Conclusion

This project demonstrated the effectiveness of kNN over Linear Regression for predicting GPS coordinates from image features. The kNN model showed superior performance, especially as the training set size increased. Additionally, feature visualization and grid search optimization provided valuable insights into model behavior and performance across various hyperparameter settings.
