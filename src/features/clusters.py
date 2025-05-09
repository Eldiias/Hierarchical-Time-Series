import pandas as pd
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import warnings
from kneed import KneeLocator
import json
import os

warnings.simplefilter("ignore")

def rolling_orders_6(x):
    """Check if average non-zero invoices in rolling windows exceeds threshold"""
    ts = x.set_index('Date').sort_index().Invoiced_Quantity
    # More efficient way to count non-zero values in each window
    return (ts > 0).rolling(6).sum().mean() > 2

def review_plant(plant_df):
    # Get customers meeting the condition
    condition1 = plant_df.groupby('customer_id').apply(rolling_orders_6)
    
    # Process Invoiced_Quantity
    plant_df_sorted = plant_df.sort_values(['customer_id', 'Date'])
    invoices = plant_df_sorted.pivot_table(
        index='Date',
        columns='customer_id',
        values='Invoiced_Quantity',
        aggfunc='first'  # or 'sum', 'mean', depending on your data
    )
    
    # Filter to customers meeting condition1
    selection1 = invoices.loc[:, condition1[condition1].index]
    # Further filter customers based on recent activity
    selection2 = selection1.loc[:, selection1.tail(6).isna().sum()!= 6]
    #(selection2.sum(axis=1)/plant_df.query('Monday_number==5').groupby(['Date']).Invoiced_Quantity.sum()).plot()
    #plt.title('comparison with demand of customers who are delivered from unique plant')
    #plt.show()
    return selection2

def plot_ts_kmeans_elbow(data, k_range=(2, 10), metric='softdtw', random_state=42):
    """
    Plot the elbow curve for TimeSeriesKMeans with specified metric.

    Parameters:
    - data: array-like, shape (n_samples, time_steps) or (n_samples, time_steps, 1)
    - k_range: tuple, (min_k, max_k) for range of clusters
    - metric: str, one of ['euclidean', 'dtw', 'softdtw']
    - random_state: int, for reproducibility
    """
    # Ensure data is 3D
    ts_data = to_time_series_dataset(data)

    inertias = []
    ks = range(k_range[0], k_range[1] + 1)

    for k in ks:
        model = TimeSeriesKMeans(n_clusters=k, metric=metric, random_state=random_state, n_jobs=-1)
        model.fit(ts_data)
        inertias.append(model.inertia_)

    kneedle = KneeLocator(
        x=ks,
        y=inertias,
        curve='convex',       # 'concave' for increasing curves
        direction='decreasing',  # or 'increasing'
        interp_method='polynomial'
    )

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker='o')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title(f"Elbow Method for TimeSeriesKMeans ({metric})")
    plt.xticks(ks)
    plt.grid(True)
    plt.show()
    return int(kneedle.knee)

def get_optimal_cluster_number_global(data):

    cluster_number = {}
    for plant, sdf in data.items():
        print(f"Processing plant: {plant}")
        ts_data = to_time_series_dataset(sdf.T.fillna(0))
        knee = plot_ts_kmeans_elbow(ts_data, k_range=(4,20))
        cluster_number[plant] = knee
    return cluster_number

def process_plant(plant, plant_df, cluster_map=None):
    print(f"Processing plant: {plant}")

    sdf = review_plant(plant_df)
    fs = to_time_series_dataset(sdf.T.fillna(0))
    ### if optimal cluster number is not known, use elbow method
    if cluster_map is None:
        knee = plot_ts_kmeans_elbow(fs, k_range=(4,25))
    ### else you can use the values from cluster_number stored in the config file.
    else:
        knee = cluster_map[plant]
    
    model = TimeSeriesKMeans(n_clusters=knee, metric="softdtw", random_state=42, n_jobs=-1)
    model.fit(fs)

    labels = {i:j for i, j in zip(sdf.columns, model.labels_)}
    selection3 = sdf.stack().reset_index(level=1)
    selection3['label'] = selection3['customer_id'].map(labels)
    return model, selection3, labels

def viz_plant(plant, labelled):
    """Visualize the clusters for a given plant"""
    print(f"Visualizing clusters for plant: {plant}")
    print('Number of clusters:', labelled[plant]['label'].nunique())
    for label in labelled[plant]['label'].unique():
        print(f'\tNumber of customers in cluster {label}:', labelled[plant][['customer_id', 'label']].drop_duplicates().label.value_counts()[label])
        labelled[plant].query("label==@label").rename(columns={0:'y'}).reset_index().set_index(['Date', 'customer_id']).y.unstack(1).plot()
        plt.legend('')
        plt.show()

def reverse_clusters(cluster_map):
    """
    Reverse the cluster mapping to get a dictionary of clusters and customers in each cluster.
    """
    reversed_map = {}
    for plant, cluster in cluster_map.items():
        for customer, label in cluster.items():
            new_label = f"{plant}_{label}"
            if new_label not in reversed_map:
                reversed_map[new_label] = []
            reversed_map[new_label].append(str(customer))
    return reversed_map

def create_hierarchical_dataset(labelled, labels, original_data, cluster_number):
    """
    Define the hierarchical structure for Hierarchical modelling
    """

    labelled_filtered = {
    i:j[j.customer_id.isin(
        j.query("Date<'01-01-2021'")
        .groupby('customer_id')[0]
        .sum().to_frame()
        .rename({0:'value'}, axis=1)
        .query('value>0').index
        )] for i,j in labelled.items()}
    
    plant_filtered = pd.concat(
    (
        j.reset_index()
        .groupby('Date')[0]
        .sum().to_frame()
        .rename({0:i}, axis=1) for i,j in labelled_filtered.items()
        ), axis=1)
    
    total_filtered = plant_filtered.sum(axis=1).to_frame().rename({0:'total'}, axis=1)
    label_filtered = pd.concat(
        (
            j.reset_index()
            .groupby(['Date', 'label'])[0]
            .sum().unstack(1)
            .rename({k:f'{i}_{k}' for k in j.label.unique()}, axis=1) for i,j in labelled_filtered.items()
            ), axis=1)
    
    customer_filtered = pd.concat(
        (i.reset_index().groupby(['Date', 'customer_id'])[0]
        .sum().unstack() for i in labelled_filtered.values()), axis=1)
    
    customer_filtered.columns = customer_filtered.columns.map(str)
    
    hierarchy_df = customer_filtered.join(label_filtered) \
                                .join(plant_filtered).join(total_filtered)
    hierarchy_df.index = pd.to_datetime(hierarchy_df.index)
    hierarchy_df = hierarchy_df.resample('MS').sum()

    print(f'Global: {total_filtered.shape[1]}')
    print(f'Number of plants: {plant_filtered.shape[1]}')
    print(f'Number of clusters: {label_filtered.shape[1]}')
    print(f'Number of customers: {customer_filtered.shape[1]}')

    total = {'total':original_data.Plant.dropna().unique().tolist()}
    plants = {i:list(map(lambda x: i+'_'+str(x),range(j))) for i,j in cluster_number.items()}
    label = reverse_clusters(labels)
    hierarchy_dict = {**total, **plants, **label}
    ddd = dict()

    for cluster_label in label.keys():
        zz = []
        for customer in hierarchy_dict[cluster_label]:
            zz.append(cluster_label + '_' + customer)
            ddd[customer] = cluster_label + '_' + customer
        hierarchy_dict[cluster_label] = zz
    
    hierarchy_df.columns = [i if i not in ddd else ddd[i] for i in hierarchy_df.columns]
    
    # split into train and test
    df_train = hierarchy_df[hierarchy_df.index<'01-01-2021'].copy()
    df_test = hierarchy_df[(hierarchy_df.index>='01-01-2021')&(hierarchy_df.index<'06-01-2022')].copy()

    return df_train, df_test, hierarchy_dict

def main():
    """
    Main function to process plant data and perform time series clustering.
    """
    print('Current working directory')
    print(os.getcwd())
    # Load data
    df = pd.read_csv('../../data/raw/plant_ids.csv')
    df.Date = pd.to_datetime(df.Date).dt.tz_localize(None)
    df = df.set_index('Date').sort_index().reset_index()

    if os.path.exists('../../src/utils/cluster_number.json'):
        with open('../../src/utils/cluster_number.json', 'r') as f:
            cluster_number = json.load(f)
    kms = dict()
    labelled = dict()
    labels = dict()
    for plant, pdf in df.groupby('Plant'):
        model, labeled, label = process_plant(plant, pdf, cluster_number)
        kms[plant] = model
        labelled[plant] = labeled
        labels[plant] = label
        
    df_train, df_test, hierarchy_dict = create_hierarchical_dataset(labelled, labels, df, cluster_number)

    df_train.to_csv('../../data/processed/hierarchy_train.csv', index=False)
    df_test.to_csv('../../data/processed/hierarchy_test.csv', index=False)

    with open('../../src/utils/hierarchy_dict.json', 'w') as f:
        json.dump(hierarchy_dict, f)
        
if __name__ == "__main__":
    main()