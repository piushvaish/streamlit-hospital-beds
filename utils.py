## for data
import numpy as np
import pandas as pd



## for geospatial
import folium
import geopy

## for machine learning
from sklearn import preprocessing

## for statistical tests
import scipy

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for deep learning
import minisom

'''
Fit a Self Organizing Map neural network.
:paramater
    :param X: dtf
    :param model: minisom instance - if None uses a map of 5*sqrt(n) x 5*sqrt(n) neurons
    :param lst_2Dplot: list - 2 features to use for a 2D plot, if None it plots only if X is 2D
:return
    model and dtf with clusters
'''
def fit_dl_cluster(X, model=None, epochs=100, lst_2Dplot=None, figsize=(10,5)):
    ## model
    model = minisom.MiniSom(x=int(np.sqrt(5*np.sqrt(X.shape[0]))), y=int(np.sqrt(5*np.sqrt(X.shape[0]))), input_len=X.shape[1]) if model is None else model
    scaler = preprocessing.StandardScaler()
    X_preprocessed = scaler.fit_transform(X.values)
    model.train_batch(X_preprocessed, num_iteration=epochs, verbose=False)
    
    ## clustering
    map_shape = (model.get_weights().shape[0], model.get_weights().shape[1])
    print("--- map shape:", map_shape, "---")
    dtf_X = X.copy()
    dtf_X["cluster"] = np.ravel_multi_index(np.array([model.winner(x) for x in X_preprocessed]).T, dims=map_shape)
    k = dtf_X["cluster"].nunique()
    print("--- found", k, "clusters ---")
    print(dtf_X.groupby("cluster")["cluster"].count().sort_values(ascending=False))
    
    ## find real centroids
    cluster_centers = np.array([vec for center in model.get_weights() for vec in center])
    closest, distances = scipy.cluster.vq.vq(cluster_centers, X_preprocessed)
    dtf_X["centroids"] = 0
    for i in closest:
        dtf_X["centroids"].iloc[i] = 1
    
    ## plot
    if (lst_2Dplot is not None) or (X.shape[1] == 2):
        lst_2Dplot = X.columns.tolist() if lst_2Dplot is None else lst_2Dplot
        utils_plot_cluster(dtf_X, x1=lst_2Dplot[0], x2=lst_2Dplot[1], th_centroids=scaler.inverse_transform(cluster_centers), figsize=figsize)

    return model, dtf_X
    
'''
Creates a map with folium.
:parameter
    :param dtf: pandas
    :param x: str - column with latitude
    :param y: str - column with longitude
    :param starting_point: list - coordinates (ex. [45.0703, 7.6869])
    :param tiles: str - "cartodbpositron", "OpenStreetMap", "Stamen Terrain", "Stamen Toner"
    :param popup: str - column with text to popup if clicked, if None there is no popup
    :param size: str - column with size variable, if None takes size=5
    :param color: str - column with color variable, if None takes default color
    :param lst_colors: list - list with multiple colors to use if color column is not None, if not given it generates randomly
    :param marker: str - column with marker variable, takes up to 7 unique values
:return
    map object to display
'''
def plot_map(dtf, x, y, start, zoom=12, tiles="cartodbpositron", popup=None, size=None, color=None, legend=False, lst_colors=None, marker=None):
    data = dtf.copy()

    ## create columns for plotting
    if color is not None:
        lst_elements = sorted(list(dtf[color].unique()))
        lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in range(len(lst_elements))] if lst_colors is None else lst_colors
        data["color"] = data[color].apply(lambda x: lst_colors[lst_elements.index(x)])

    if size is not None:
        scaler = preprocessing.MinMaxScaler(feature_range=(3,15))
        data["size"] = scaler.fit_transform(data[size].values.reshape(-1,1)).reshape(-1)

    ## map
    map_ = folium.Map(location=start, tiles=tiles, zoom_start=zoom)

    if (size is not None) and (color is None): 
        data.apply(lambda row: folium.CircleMarker(location=[row[x],row[y]], popup=row[popup],
                                                   color='#3186cc', fill=True, radius=row["size"]).add_to(map_), axis=1)
    elif (size is None) and (color is not None):
        data.apply(lambda row: folium.CircleMarker(location=[row[x],row[y]], popup=row[popup],
                                                   color=row["color"], fill=True, radius=5).add_to(map_), axis=1)
    elif (size is not None) and (color is not None):
        data.apply(lambda row: folium.CircleMarker(location=[row[x],row[y]], popup=row[popup],
                                                   color=row["color"], fill=True, radius=row["size"]).add_to(map_), axis=1)
    else:
        data.apply(lambda row: folium.CircleMarker(location=[row[x],row[y]], popup=row[popup],
                                                   color='#3186cc', fill=True, radius=5).add_to(map_), axis=1)
    
    ## legend
    if (color is not None) and (legend is True):
        legend_html = """<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">&nbsp;<b>"""+color+""":</b><br>"""
        for i in lst_elements:
            legend_html = legend_html+"""&nbsp;<i class="fa fa-circle fa-1x" style="color:"""+lst_colors[lst_elements.index(i)]+""""></i>&nbsp;"""+str(i)+"""<br>"""
        legend_html = legend_html+"""</div>"""
        map_.get_root().html.add_child(folium.Element(legend_html))
    
    ## add marker
    if marker is not None:
        lst_elements = sorted(list(dtf[marker].unique()))
        lst_colors = ["orange","orange","orange"]  #9
        ### too many values, can't mark
        if len(lst_elements) > len(lst_colors):
            raise Exception("marker has uniques > "+str(len(lst_colors)))
        ### binary case (1/0): mark only 1s
        elif len(lst_elements) == 2:
            data[data[marker]==lst_elements[1]].apply(lambda row: folium.Marker(location=[row[x],row[y]], popup=row[marker], draggable=False, 
                                                                                icon=folium.Icon(color=lst_colors[0])).add_to(map_), axis=1) 
        ### normal case: mark all values
        else:
            for i in lst_elements:
                data[data[marker]==i].apply(lambda row: folium.Marker(location=[row[x],row[y]], popup=row[marker], draggable=False, 
                                                                      icon=folium.Icon(color=lst_colors[lst_elements.index(i)])).add_to(map_), axis=1)
    return map_


def label_utilization(row):
    if row['BED_UTILIZATION'] <= 0.33:
        return 'low'
    elif (row['BED_UTILIZATION'] > 0.33) & (row['BED_UTILIZATION'] < 0.66):
        return 'medium'
    else:
        return 'high'