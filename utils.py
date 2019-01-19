import time
import warnings
from itertools import cycle, islice

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def fibs(n):
    a = 1
    b = 1
    while b < n:
        yield b
        a, b = b, a + b
        
def prepare_dataset(X, y, params):
    return ((X, y), params)

def permutate_dataset(X, num, scale=0.1, seed=42):
    l, f = X.shape
    assert num <= l
    np.random.seed(seed)
    val = np.random.normal(loc=0.0, scale=scale, size=(1,2))
    X[np.random.randint(0, l,size=num),:] += val 
    np.random.seed()
    return X

def modify_dataset(dataset, labels, num=0, scale=0.1, seed=42):
    assert num <= dataset.shape[0]
    result = []
    for f_num in iter(fibs(num)):
        result.append((permutate_dataset(np.copy(dataset), f_num, scale, seed), np.copy(labels)))
    return result

                      
def trfm_dataset(dataset, labels, action, seed=42):
    result = []
    for T in iter(get_trfm_mats(action, seed)):
        result.append((np.dot(dataset, T), np.copy(labels)))
    return result
                      
def get_trfm_mats(trfm_type='c', seed=None):
    assert trfm_type == 'c' or trfm_type == 's'
    
    orig = np.array([[1,0],[0,1]], dtype=np.float32)
    np.random.seed(seed)
    T = np.random.randint(1,101, (2,2))/100 if trfm_type == 'c' else np.random.randint(100,201, (2,2))/100
    np.random.seed()
    
    t1 = np.copy(orig)
    t1[0,0] = T[0,0]
    
    t2 = np.copy(orig)
    t2[1,1] = T[1,1]
    
    t3 = np.copy(orig)
    t3[0,0] = T[0,0]
    t3[1,1] = T[1,1]
    
    t4 = np.copy(orig)
    t4[0,1] = T[0,1]
    
    t5 = np.copy(orig)
    t5[1,0] = T[1,0]
    
    t6 = np.copy(orig)
    t6[0,1] = T[0,1]
    t6[1,0] = T[1,0]
    
    t7 = np.copy(T)
    t7[1,0] = 0
    
    t8 = np.copy(T)
    t8[0,1] = 0
    
    return orig,t1,t2,t3,t4,t5,t6,t7,t8,T


def get_squares(n_samples=500, n_clusters=3, std=[1,1,1], center_box=(-5,5), seed=42):
    assert len(std) == n_clusters
    n_samples = int(n_samples/n_clusters)
    X, y = datasets.make_blobs(n_clusters, 2, n_clusters, center_box=center_box, random_state=seed)
    data = std[0]*np.random.rand(n_samples,2)+X[0,:]
    labels = [0]*n_samples
    for i in range(1, n_clusters):
        data = np.append(data, (std[i]*np.random.rand(n_samples,2)+X[i,:]), axis=0)
        labels = np.append(labels, [i]*n_samples)
    return data, labels

def get_circles(n_samples=500, n_clusters=3, cov=[1,1,1], center_box=(-5,5), seed=42):
    assert len(cov) == n_clusters
    n_samples = int(n_samples/n_clusters)
    X, y = datasets.make_blobs(n_clusters, 2, n_clusters, center_box=center_box)
    data, labels = get_gaussian_circles(cov[0], 2*n_samples, 2, mean=X[0,:], inner_idx = 0, seed=seed)
    for i in range(1, n_clusters):
        ggc = get_gaussian_circles(cov[i], 2*n_samples, 2 ,mean=X[i,:],inner_idx=0,seed=seed)
        data = np.append(data, ggc[0], axis=0)
        labels = np.append(labels, ggc[1]+i)
    return data, labels

def get_gaussian_circles(cov, n_samples, n_classes, mean=None, inner_idx=1, seed=42):
    assert inner_idx < n_classes
    X, y = datasets.make_gaussian_quantiles(mean, cov=cov, n_samples=n_samples, n_classes=n_classes, random_state=seed)
    if inner_idx > 0: return X[y%inner_idx==0, :], y[y%inner_idx==0]/inner_idx
    else: return X[y==0, :], y[y==0]

def plot_gaussian_circles(X, y, trns_matrix=[[1,0],[0, 1]]):
    n_classes = len(np.unique(y)) + 1
    data = np.dot(X,np.array(trns_matrix))
    
    plt.figure(figsize=(n_classes*5,5))
    for class_idx in range(0, n_classes-1):
        plt.subplot(1, n_classes, class_idx+1)
        plt.scatter(data[y==class_idx, 0], data[y==class_idx, 1], s=5, c=y[y==class_idx])
        
    plt.subplot(1, n_classes, n_classes)
    plt.scatter(data[:, 0], data[:, 1], s=5, c=y)
    plt.show()

def clustering(datasets, standardize=True, pic_name=None, draw=False): 
    fig = plt.figure(figsize=(24, len(datasets)*6))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.1, hspace=.15)

    plot_num = 1
    df = []
    default_base = {'eps': .3, 'n_clusters': 3, 'dataset_name':'no_name'}
    for i_dataset, (dataset, algo_params) in enumerate(datasets):

        # update parameters with dataset-specific values
        params = default_base.copy()
        params.update(algo_params)

        X, y_gt = dataset

        # normalize dataset for easier parameter selection
        if standardize:
            X = StandardScaler().fit_transform(X)

        # ============
        # Create cluster objects
        # ============

        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
        spectral=cluster.SpectralClustering(n_clusters=params['n_clusters'],eigen_solver='arpack',
                                            affinity="nearest_neighbors")
        dbscan = cluster.DBSCAN(eps=params['eps'])
        gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
        clustering_algorithms = (
            ('MiniBatchKMeans', two_means),
            ('SpectralClustering', spectral),
            ('DBSCAN', dbscan),
            ('GaussianMixture', gmm)
        )

        for name, algorithm in clustering_algorithms:
            
            t0 = time.time()
            algorithm.fit(X)
            t1 = time.time()

            if hasattr(algorithm, 'labels_'): y_pred = algorithm.labels_.astype(np.int)
            else: y_pred = algorithm.predict(X)
            
            h, c, vm = metrics.homogeneity_completeness_v_measure(y_gt, y_pred)
            db_score = metrics.davies_bouldin_score(X, y_pred) if len(np.unique(y_pred)) > 1 else None
            
            df_data = {
                'homogeneity':h,
                'completeness':c,
                'v_measure':vm,
                'davies_bouldin_score':db_score,
                'time':t1-t0}
            index = pd.MultiIndex.from_tuples([(params['dataset_name'], name)], names=['dataset', 'algorithm'])
            df.append(pd.DataFrame(df_data, index=index))

            plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
            if i_dataset == 0: plt.title("{0} \n {1}".format(name, params['dataset_name']), size=18)
            else: plt.title(params['dataset_name'], size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            #plt.xlim(-2.5, 2.5)
            #plt.ylim(-2.5, 2.5)
            #plt.xticks(())
            #plt.yticks(())
            plt.text(.99,.01,('%.2fs'%(t1-t0)).lstrip('0'),transform=plt.gca().transAxes,size=15,horizontalalignment='right')
            plot_num += 1
    
    pic_name = pic_name if pic_name else str(time.time())
    fig.savefig(pic_name+'.png')
    if draw:plt.show()
    else : plt.close(fig)
    
    return pd.concat(df)