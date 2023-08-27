from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import pickle
import os

np.random.seed(42)

def generate_cluster(center, num_points, spread=0.1):
    """Generate points around a center."""
    return center + np.random.randn(num_points, 2) * spread

def assign_id_with_error(true_id, total_clusters, error_rate):
    """Assign a cluster ID, but with a given error rate."""
    if np.random.rand() < error_rate:
        return np.random.choice([i for i in range(total_clusters) if i != true_id])
    return true_id

def generate_clusters(num_clusters, num_points, spread=0.1, error_rate=0.1):
    """Generate multiple clusters of points in 2D."""
    centers = np.random.rand(num_clusters, 2) * 10  # for example in a 10x10 grid

    points = []
    true_labels = []
    noisy_labels = []

    for i, center in enumerate(centers):
        cluster_points = generate_cluster(center, num_points, spread)
        cluster_true_labels = [i] * num_points
        cluster_noisy_labels = [assign_id_with_error(i, num_clusters, error_rate) for _ in range(num_points)]

        points.extend(cluster_points)
        true_labels.extend(cluster_true_labels)
        noisy_labels.extend(cluster_noisy_labels)

    return np.array(points), np.array(true_labels), np.array(noisy_labels)

def generate_data():
    points, true_labels, noisy_labels = generate_clusters(4, 500, spread=1, error_rate=0.01)
    df = pd.DataFrame(points, columns=['x', 'y'])
    df['label'] = noisy_labels
    return df


with open("dataset_1model_1.pkl", "wb") as f:
    pickle.dump(generate_data(), f)


def get_data(data_name):
    data = None
    with open(f"{data_name}.pkl", "rb") as f:
        data = pickle.load(f)
    return data

def exist_data(data_name):
    return os.path.exists(f"{data_name}.pkl")
# -------------------------------------------------


class KNN_Model:
    def predict(data, model_args=None):
        model_args = model_args if model_args else {}
        model = KNeighborsClassifier(**model_args)
        X = data[['x', 'y']]
        y = data['label']
        model.fit(X, y)
        return model.predict(X)

class DBSCAN_Model:
    def predict(data, model_args=None):
        model_args = model_args if model_args else {}
        model = DBSCAN(**model_args)
        return model.fit_predict(data)

CLUSTER_MODELS = {
    "K-NN": KNN_Model,
    "DBSCAN": DBSCAN_Model
}

class Cluster(BaseModel):
    def __init__(self, model_name, data_name, cluster_model_name, cluster_model_args):
        self.model_name = model_name
        self.data_name = data_name
        self.cluster_model_name = cluster_model_name
        self.cluster_model_args = cluster_model_args
        self.cluster_model = CLUSTER_MODELS[self.cluster_model_name]
        self.df = None

        self.load_data()

    def load_data(self):
        self.df = get_data(f"{self.data_name}{self.model_name}") # load reduced embeddings into pandas as x, y, and label
        exist_cluster_file = exist_data(f"{self.cluster_model}{self.data_name}{self.model_name}") # determine if clusters have been calculated
        if not exist_cluster_file:
            self.df["cluster"] = self.generate_clusters()
        else:
            self.df["cluster"] = get_data(f"{self.cluster_model}{self.data_name}{self.model_name}")
        self.save()


    def generate_clusters(self):#
        return self.cluster_model.predict(self.df, self.cluster_model_args)

    def save(self):
        with open(f"{self.cluster_model_name}{self.data_name}{self.model_name}", "wb") as f:
            pickle.dump(self.df["cluster"], f)

    def get_ids(self, ids):
        pass # get ids from cluster

    def get_error(self, ids):
        temp_data = self.df.copy()
        clusters = temp_data['cluster'].unique()

        # Assign the most common label within each cluster as the "cluster label"
        cluster_labels = temp_data.groupby('cluster')['label'].apply(lambda x: x.mode().iloc[0])

        # 1. Determine the Confidence of Each Cluster
        cluster_confidence = temp_data.groupby('cluster').apply(lambda x: (x['label'] == cluster_labels[x.name]).mean())

        # 2. Calculate Mean Location for Each Cluster
        cluster_centers = temp_data.groupby('cluster')[['x', 'y']].mean().to_dict(orient='index')

        # 3. Determine the Relative Position of Each Point
        temp_data['dist_to_centroid'] = temp_data.apply(lambda row: np.linalg.norm(np.array([row['x'], row['y']]) - np.array(
            [cluster_centers[row['cluster']]['x'], cluster_centers[row['cluster']]['y']])), axis=1)
        max_dists = temp_data.groupby('cluster')['dist_to_centroid'].transform('max')
        temp_data['relative_dist'] = temp_data['dist_to_centroid'] / max_dists

        # 4. Calculate Error Probability for Each Point
        temp_data['error_prob'] = np.where(temp_data['label'] == temp_data['cluster'].map(cluster_labels),
                                    (1 - temp_data['cluster'].map(cluster_confidence)) * temp_data['relative_dist'],
                                    temp_data['cluster'].map(cluster_confidence) * (1 - temp_data['relative_dist']))

        # Return the error probability for the given point IDs
        error_probs = temp_data.loc[temp_data.index.isin(ids), 'error_prob'].to_dict()

        return error_probs

    def update(self):
        self.generate_clusters()
        self.save()
