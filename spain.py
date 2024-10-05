"""
The goal of this assignment is to predict GPS coordinates from image features using k-Nearest Neighbors.
Specifically, have featurized 28616 geo-tagged images taken in Spain split into training and test sets (27.6k and 1k).

The assignment walks students through:
    * visualizing the data
    * implementing and evaluating a kNN regression model
    * analyzing model performance as a function of dataset size
    * comparing kNN against linear regression

Images were filtered from Mousselly-Sergieh et al. 2014 (https://dl.acm.org/doi/10.1145/2557642.2563673)
and scraped from Flickr in 2024. The image features were extracted using CLIP ViT-L/14@336px (https://openai.com/clip/).
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def plot_data(train_feats, train_labels):
    """
    Input:
        train_feats: Training set image features
        train_labels: Training set GPS (lat, lon)

    Output:
        Displays plot of image locations, and first two PCA dimensions vs longitude
    """
    # Create a 'plots' folder if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Plot image locations
    plt.scatter(train_labels[:, 1], train_labels[:, 0], marker=".")
    plt.title('Image Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('plots/image_locations.png')
    plt.close()

    # Run PCA on training_feats
    ##### TODO(a): Your Code Here #####
    transformed_feats = StandardScaler().fit_transform(train_feats)
    transformed_feats = PCA(n_components=2).fit_transform(transformed_feats)

    # Plot images by first two PCA dimensions
    plt.scatter(transformed_feats[:, 0],
                transformed_feats[:, 1],
                c=train_labels[:, 1],
                marker='.')
    plt.colorbar(label='Longitude')
    plt.title('Image Features by Longitude after PCA')
    plt.savefig('plots/pca_features.png')
    plt.close()


def grid_search(train_features, train_labels, test_features, test_labels, is_weighted=False, verbose=True):
    """
    Input:
        train_features: Training set image features
        train_labels: Training set GPS (lat, lon) coords
        test_features: Test set image features
        test_labels: Test set GPS (lat, lon) coords
        is_weighted: Weight prediction by distances in feature space

    Output:
        Prints mean displacement error as a function of k
        Plots mean displacement error vs k

    Returns:
        Minimum mean displacement error
    """
    # Evaluate mean displacement error (in miles) of kNN regression for different values of k
    # Technically we are working with spherical coordinates and should be using spherical distances, but within a small
    # region like Spain we can get away with treating the coordinates as cartesian coordinates.

    knn = NearestNeighbors(n_neighbors=100).fit(train_features)

    if verbose:
        print(f'Running grid search for k (is_weighted={is_weighted})')

    ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
    mean_errors = []
    for k in ks:
        distances, indices = knn.kneighbors(test_features, n_neighbors=k)

        errors = []
        for i, nearest in enumerate(indices):
            # Evaluate mean displacement error in miles for each test image
            # Assume 1 degree latitude is 69 miles and 1 degree longitude is 52 miles
            y = test_labels[i]

            ##### TODO(d): Your Code Here #####
            if is_weighted: 
                weights = 1.0 / (distances[i] + 1e-8)  # Avoid division by zero with small epsilon
                weighted_lat = np.average(train_labels[nearest, 0], weights=weights)
                weighted_lon = np.average(train_labels[nearest, 1], weights=weights)

                test_lat = test_labels[i][0]
                test_lon = test_labels[i][1]
                
                # Convert differences in lat and long to miles
                lat_diff = (weighted_lat - test_lat) * 69
                lon_diff = (weighted_lon - test_lon) * 52
            
               
            else: 
                avg_lat = np.mean(train_labels[nearest, 0])
                avg_lon = np.mean(train_labels[nearest, 1])
                test_lat = test_labels[i][0]
                test_lon = test_labels[i][1]
                
                # Convert differences in lat and long to miles
                lat_diff = (avg_lat - test_lat) * 69
                lon_diff = (avg_lon - test_lon) * 52
            
            # Test image coordinates
            
            # Euclidean distance in miles
            displacement_error = np.sqrt(lat_diff**2 + lon_diff**2)

            errors.append(displacement_error)
        
        e = np.mean(np.array(errors))
        mean_errors.append(e)
        if verbose:
            print(f'{k}-NN mean displacement error (miles): {e:.1f}')

    # Plot error vs k for k Nearest Neighbors
    if verbose:
        plt.plot(ks, mean_errors)
        plt.xlabel('k')
        plt.ylabel('Mean Displacement Error (miles)')
        plt.title('Mean Displacement Error (miles) vs. k in kNN')
        plt.savefig(f'plots/knn_error_{"weighted" if is_weighted else "unweighted"}.png')
        plt.close()

        min_error_index = np.argmin(mean_errors)

        # k values corresponding to the mean_errors
        k_values = np.arange(1, len(mean_errors) + 1)

        # Print k with the lowest mean error
        print("The k with the lowest mean error is:", k_values[min_error_index])

    return min(mean_errors)


def main():
    print("Predicting GPS from CLIP image features\n")

    # Import Data
    print("Loading Data")
    data = np.load('im2spain_data.npz')

    train_features = data['train_features']  # [N_train, dim] array
    test_features = data['test_features']    # [N_test, dim] array
    train_labels = data['train_labels']      # [N_train, 2] array of (lat, lon) coords
    test_labels = data['test_labels']        # [N_test, 2] array of (lat, lon) coords
    train_files = data['train_files']        # [N_train] array of strings
    test_files = data['test_files']          # [N_test] array of strings

    # Data Information
    print('Train Data Count:', train_features.shape[0])

    # Part A: Feature and label visualization (modify plot_data method)
    plot_data(train_features, train_labels)

    # Part B: Find the 3 nearest neighbors of test image 53633239060.jpg
    knn = NearestNeighbors(n_neighbors=3).fit(train_features)

    # Use knn to get the k nearest neighbors of the features of image 53633239060.jpg
    ##### TODO(c): Your Code Here #####
    k_indx = np.where(test_files == '53633239060.jpg')[0]
    features_selected = test_features[k_indx]
    test_coords = test_labels[k_indx]

    # Find the three nearest neighbors
    distances, indices = knn.kneighbors(features_selected, n_neighbors=3, return_distance=True)

    print(distances, indices)
    # Retrieve the coordinates of the three nearest neighbors
    neighbor_coords = [train_labels[idx] for idx in indices[0]]
    print("Neighbors", neighbor_coords)
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Create a 2x2 grid of subplots
    axs = axs.flatten()  # Flatten the 2D array of axes to 1D

    # Plot the test image location on the first subplot with all data points
    axs[0].scatter(train_labels[:, 1], train_labels[:, 0], marker=".", label='All Training Images')
    axs[0].scatter(test_coords[0][1], test_coords[0][0], color='blue', s=100, label='Test Image', edgecolors='black')
    axs[0].set_title('Test Image Location')
    axs[0].set_xlabel('Longitude')
    axs[0].set_ylabel('Latitude')
    axs[0].legend()

    # Plot each neighbor on separate subplots with all data points
    for i, coords in enumerate(neighbor_coords):
        axs[i+1].scatter(train_labels[:, 1], train_labels[:, 0], marker=".", label='All Training Images')
        axs[i+1].scatter(coords[1], coords[0], color='red', s=100, label=f'Neighbor {i+1}', edgecolors='black')
        axs[i+1].set_title(f'Neighbor {i+1} Location')
        axs[i+1].set_xlabel('Longitude')
        axs[i+1].set_ylabel('Latitude')
        axs[i+1].legend()

    plt.tight_layout()
    plt.savefig('plots/nearest_neighbors.png')
    plt.close()

    c = []
    for idx in indices: 
        c.append(train_files[idx])
    print("here it is: ", c)

    
    # Part D: establish a naive baseline of predicting the mean of the training set
    ##### TODO(d): Your Code Here #####
    centroid_lat = np.mean(train_labels[:, 0])
    centroid_long = np.mean(train_labels[:, 1])
    centroid = (centroid_lat, centroid_long)

    print("CENTROID", centroid)
    def calculate_distance(lat1, long1, lat2, long2):
        lat_miles = (lat1 - lat2) * 69
        long_miles = (long1 - long2) * 52
        return np.sqrt(lat_miles**2 + long_miles**2)

    errors = [calculate_distance(test_lat, test_long, centroid[0], centroid[1]) for test_lat, test_long in test_labels]
    mde = np.mean(errors)
    
    print(f"Mean Displacement Error (MDE) for the constant baseline is: {mde:.2f} miles")



    # Part E: complete grid_search to find the best value of k
    grid_search(train_features, train_labels, test_features, test_labels)
    #best value = 3

    # Parts G: rerun grid search after modifications to find the best value of k
    grid_search(train_features, train_labels, test_features, test_labels, is_weighted=True)

    # Part H: compare to linear regression for different # of training points
    def compute_error(y_true, y_pred):

        lat_diff = (y_pred[:, 0] - y_true[:, 0]) * 69
        lon_diff = (y_pred[:, 1] - y_true[:, 1]) * 52
        return np.sqrt(lat_diff**2 + lon_diff**2)


    mean_errors_lin = []
    mean_errors_nn = []
    ratios = np.arange(0.1, 1.1, 0.1)
    for r in ratios:

        
        num_samples = int(r * len(train_features))

        e_nn = grid_search(train_features[:num_samples], train_labels[:num_samples],
                                                    test_features, test_labels,
                                                    is_weighted=True, verbose=False)

        ##### TODO(h): Your Code Here #####
        lin_reg = LinearRegression()
        lin_reg.fit(train_features[:num_samples], train_labels[:num_samples])
        lin_reg_pred = lin_reg.predict(test_features)
        e_lin = np.mean(compute_error(test_labels, lin_reg_pred))

        mean_errors_lin.append(e_lin)
        mean_errors_nn.append(e_nn)

        print(f'\nTraining set ratio: {r} ({num_samples})')
        print(f'Linear Regression mean displacement error (miles): {e_lin:.1f}')
        print(f'kNN mean displacement error (miles): {e_nn:.1f}')

    # Plot error vs training set size
    plt.plot(ratios, mean_errors_lin, label='lin. reg.')
    plt.plot(ratios, mean_errors_nn, label='kNN')
    plt.xlabel('Training Set Ratio')
    plt.ylabel('Mean Displacement Error (miles)')
    plt.title('Mean Displacement Error (miles) vs. Training Set Ratio')
    plt.legend()
    plt.savefig('plots/error_vs_training_size.png')
    plt.close()
       

if __name__ == '__main__':
    main()