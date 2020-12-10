import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.cluster import KMeans as kmeans
from numpy.linalg import eig
from sklearn.cluster import SpectralClustering
from scipy.linalg import eig as sp_eig

####### k_means clustering method ##############
def k_means(k_center, data_points):

    cluster_table = data_points
    data_length = len(data_points)
    k = random.sample(range(data_length), k_center)
    center_j = data_points.iloc[k] # This is the k center points

    for n in range(500):

        ######## Find the ########
        m_ij = pd.DataFrame(0, index= np.arange(data_length), columns=k)
        min_search = pd.Series(data= 0, index=k)
        ####### Assign each point to their nearest center. The columns of m_ij reprent each center #######
        for i in range(len(data_points)):
            min_search = pd.Series(data=0, index=k)
            for key, value in data_points.iteritems():
                min_search = min_search + (center_j[key] - data_points[key].iloc[i]).pow(2)
            m_ij[min_search.idxmin(axis = 1)].iloc[i] = 1
        ####### Normalize each data_center matrix m_ij, which can be the used to update the center location on next iteration #######
        for i in range(len(k)):
            if m_ij[k[i]].sum() != 0:
                m_ij[k[i]] = m_ij[k[i]]/m_ij[k[i]].sum()
        end_number = center_j - m_ij.T.dot(cluster_table)

        ####### If there is no changes in the iteration, stop running#######
        if end_number.sum().sum() == 0:
            break
        ####### Update the center #######
        center_j = m_ij.T.dot(data_points)
    m_ij.columns = np.arange(k_center)
    Label = m_ij.idxmax(axis = 1)

    #
    # # Reset the index for m_ij and all cluster centers in orders
    # m_ij[m_ij != 0] = 1
    # center_j.set_index(np.arange(k_center), inplace = True)

    return Label


####### diameter clustering method ##############

def diameter_cluster(k_center, data_points):
    data_length = len(data_points)
    cluster_table = data_points[[0,1]]
    center = pd.DataFrame(0, index= np.arange(k_center), columns= np.arange(2))
    k = random.sample(range(data_length), 1)
    ####### If there is only one center, return the data_points' label as all 0 #######
    if k_center <= 1:
        center.iloc[0] = cluster_table.iloc[k[0]]
        data_points[2] = 0
        Label = data_points[2]
        return Label
    ####### If there is more than one center, random pick the first one center #######
    else:
        center.iloc[0] = cluster_table.iloc[k[0]]
    ####### Find the furthest nearest point to all i-1 th center to be the i th center #######
        for i in range(1, k_center):
            max_search = pd.DataFrame(0, index= np.arange(data_length), columns= np.arange(i))
            for j in range(i):
                for key, value in cluster_table.iteritems():
                    max_search[j] = max_search[j] + (cluster_table[key] - center[key].iloc[j]).pow(2)

            m_ij = max_search.min(axis= 1)
            center.iloc[i] = cluster_table.iloc[m_ij.idxmax(axis = 1)]

        min_search = pd.DataFrame(0, index=np.arange(data_length), columns=np.arange(k_center))
        for i in range(k_center):
            for key, value in cluster_table.iteritems():
                min_search[i] = min_search[i] + (cluster_table[key] - center[key].iloc[i]).pow(2)

            m_ij = min_search.idxmin(axis=1)
        Label = m_ij
        return Label


def spectral_clustering(k_center, data_points):
    cluster_table = data_points[[0, 1]]
    ####### Calculate the Adjacency Matrix A #######
    A_ij = pd.DataFrame(0, index=np.arange(len(cluster_table)), columns=np.arange(len(cluster_table)))
    for i in range(len(cluster_table)):
        for key, value in cluster_table.iteritems():
            A_ij[i] = A_ij[i] + (cluster_table[key] - cluster_table[key].iloc[i]).pow(2)
    A_ij = np.exp(-0.1 * A_ij) # Apply Gaussian kernel with an arbitrary coefficient
    A_ij[A_ij==1] = 0
    ####### Apply the K nearest neighbour with k =5 to decide the edges #######
    for i in range(len(cluster_table)):
        for j in range(4):
            A_ij[i].iloc[A_ij[i].idxmax()] = -1
    A_ij[A_ij != -1] = 0
    A_ij[A_ij == -1] =1
    A_ij = A_ij +A_ij.T
    A_ij[A_ij > 0] = 1
    A_ij.to_csv('A_ij.csv')

    ####### Calculate the Degree Matrix #######
    D_matrix = pd.DataFrame(0, index=np.arange(len(cluster_table)), columns=np.arange(len(cluster_table)))
    for i in range(len(cluster_table)):
        k = A_ij.iloc[i].sum()
        D_matrix[i].iloc[i] = int(k)

    ####### Calculate the Laplacian Matrix #######
    L_matrix = D_matrix - A_ij
    L_matrix.to_csv('L_matrix.csv')

    ####### Calculate the eigenvalue and eigenvector #######
    w, v = sp_eig(L_matrix)
    pd_w = pd.Series(w.real)
    pd_v = pd.DataFrame(v.real)
    pd_w.to_csv('w.csv')
    pd_v.to_csv('v.csv')

    ####### Find the eigenvectors corresponding to the smallest k eigenvalues #######
    search_w = pd_w
    index = pd.Series(0, index=np.arange(k_center))
    for i in range(k_center):
        index[i] = search_w.idxmin(axis=0)
        search_w[index[i]] =999
    eigenvectors = pd.DataFrame(v.real)
    eigenvectors_first_k = eigenvectors[index]
    eigenvectors_first_k.to_csv('eigen.csv')

    ####### Run k-means of the rows of the eigenvectors matrix and return the cluster label #######
    # Label = k_means(k_center, eigenvectors_first_k)
    ####### The sklearn package k-means method is used here #######
    Label = kmeans(n_clusters=k_center).fit(eigenvectors_first_k).labels_
    return Label


n_clusters = [2, 2, 3]
method = ["Algorithm_1_K-means", "Algorithm_2_Diameter_Clustering", "Algorithm_3_Spectral Clustering"]
for i in range(1):

    data = pd.read_csv('Dataset_'+str(i+1)+'.csv', header=None)
    area = 10
    fig = plt.figure(0)
    plt.scatter(data[0], data[1], s=area, c=data[2], alpha=1)
    fig.suptitle('Dataset_'+str(i+1)+"_Original_Labeled", fontsize=15)
    fig.savefig('Dataset_'+str(i+1)+"_Original_Labeled")

    Label_0 = k_means(n_clusters[i], data)
    area = 10
    fig = plt.figure(1)
    plt.scatter(data[0], data[1], s=area, c=Label_0, alpha=1)
    fig.suptitle('Dataset_'+str(i+1)+"_"+method[0], fontsize=15)
    fig.savefig('Dataset_'+str(i+1)+"_"+method[0])

    Label_1 = diameter_cluster(n_clusters[i], data)
    #print(Label_1)
    area = 10
    fig = plt.figure(2)
    plt.scatter(data[0], data[1], s=area, c=Label_1, alpha=1)
    fig.suptitle('Dataset_'+str(i+1)+"_"+method[1], fontsize=15)
    fig.savefig('Dataset_'+str(i+1)+"_"+method[1])


    data_2 = data[[0,1]]
    Label_2 = spectral_clustering(n_clusters[i], data)
    # clustering = SpectralClustering(n_clusters=n_clusters[i], assign_labels = "discretize", random_state = 0).fit(data_2)
    # Label_2 = clustering.labels_
    area = 10
    fig = plt.figure(3)
    plt.scatter(data[0], data[1], s=area, c=Label_2, alpha=1)
    fig.suptitle('Dataset_'+str(i+1)+"_"+method[2], fontsize=15)
    fig.savefig('Dataset_'+str(i+1)+"_"+method[2])

    plt.show()
