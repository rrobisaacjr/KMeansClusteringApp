# K-Means Clustering Program

This program implements K-Means clustering to classify feature vectors using a dataset containing information about the chemical content of wine.

## Usage

1. Run the program by executing `python kmeans_clustering.py`.
2. Select two columns of data from the provided options.
3. Enter the desired number of clusters.
4. Click the "Run" button to perform the clustering.

## Functionality

- **Data Selection:** Users can choose two columns of data from the available options in the dropdown menus.
- **Cluster Selection:** Users specify the number of clusters to classify the data into.
- **Output Generation:** The program generates an output file named `output.csv`, containing the centroids and the points under each centroid.
- **Visualization:** A scatterplot graph is displayed, where each color represents a cluster (maximum of 10 clusters).
- **Centroid Analysis:** The program displays the centroids and the points under each centroid in a scrollable list box.

## Files

- `kmeans_clustering.py`: Main Python script implementing the K-Means clustering functionality.
- `wine.csv`: Dataset containing information about the chemical content of wine.
- `output.csv`: Output file containing the centroids and the points under each centroid.

## Dependencies

- Python 3.x
- pandas
- numpy
- tkinter
- matplotlib

## The Algorithm: K-Means Clustering

The K-Means algorithm is a straightforward method for partitioning a dataset into clusters.

### Initialization of Centroids:

- Begin by randomly initializing k centroids. Each centroid represents a cluster.
- Typically, k observations are chosen randomly from the dataset to serve as initial centroids.

### Assignment of Data Points to Clusters:

- For each data point, calculate the distance to each centroid using a distance metric, such as Euclidean distance.
- Assign each data point to the nearest centroid, thereby forming k clusters. The distance metric is typically calculated using the formula:

```math
 d = \sqrt{\sum_{i=0}^{m-1}(x_i - c_i)^2}
```

where:
- \( x \) is the current feature vector being classified,
- \( c \) is the current centroid to which \( x \)'s distance is being computed,
- \( m \) is the feature vector length.


### Update of Centroids:

To update centroids, the program calculates the average coordinates of the feature vectors. For instance, if the feature vector has a length of 2, the update equation would be:

```math
c_i = \left( \frac{x_0 + x_1 + x_2 + ...}{\text{count}(class\ i)}, \frac{y_0 + y_1 + y_2 + ...}{\text{count}(class\ i)} \right)
```

Similarly, for a feature vector length of 3, the update equation becomes:

```math
c_i = \left( \frac{x_0 + x_1 + x_2 + ...}{\text{count}(class\ i)}, \frac{y_0 + y_1 + y_2 + ...}{\text{count}(class\ i)}, \frac{z_0 + z_1 + z_2 + ...}{\text{count}(class\ i)} \right)
```

This iterative update process continues until the centroids stabilize or a predetermined number of iterations is reached. It ensures that the centroids accurately represent the centers of the clusters and that data points are correctly assigned to their respective clusters based on proximity.

