import os
import warnings
from itertools import cycle, islice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid
from analisis_tiempo import cargar_tiempo

nombres = []
pd.options.mode.chained_assignment = None  # default='warn'


def cargar_directorio(ruta):
    # Cargo todos los archivos que empiecen por 'TE'
    dataDir = ruta
    mats = []
    directorio_todo = []
    i = 0

    if not os.path.isdir(ruta):
        raise EntradaError("Error en la ruta")

    for file in os.listdir(dataDir):
        cambio = False
        siguiente = []
        mats.append(loadmat(dataDir + file))
        array = directorio_todo
        # Cargo todos los nombres de los archivos del directorio
        file = file.replace(".mat", "")
        nombres.append(file)
        # siguiente = next(v for k, v in mats[i].items() if 'TE' in k)
        for k, v in mats[i].items():
            if 'TE' in k:
                siguiente = v
                cambio = True
            elif 'ET' in k:
                siguiente = v
                cambio = True
            elif 'ID' in k:
                siguiente = v
                cambio = True
        if len(array) != 0:
            directorio_todo = np.concatenate((array, siguiente), axis=0)
        else:
            directorio_todo = siguiente
        i += 1
        if not cambio:
            raise EntradaError("Error en la estructura de los datos aportados")
    return directorio_todo


def get_nombres():
    return nombres


def cargar_espiral(siguiente, file_name):
    puntos_todos = []
    vector_tiempos = []
    nombre_enDataframe = []
    existe = False
    # Cargo todas las espirales del directorio
    for each_file in siguiente:
        nombre_enDataframe.append(each_file['sname'][0])
        puntos_todos.append(each_file['thePoints'][0])
        vector_tiempos.append(each_file['Ts'][0])
    i = 0
    espiral_pedida = pd.DataFrame()
    while i < len(puntos_todos):
        # cargo una espiral si coincide con el nombre aportado
        # if (numero_espiral == i and nombres[i] == file_name) or numero_espiral == -1:

        if nombres[i] == file_name:
            existe = True
            # Primero comprobamos que el nombre en el dataframe se parezca al menos un 50% para asegurarnos
            # de que la espiral escogida es la correcta
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, nombre_enDataframe[i][0], nombres[i]).ratio()
            if ratio > 0.4:
                # espiral
                df1 = pd.DataFrame(puntos_todos[i].astype(float), columns=['x', 'y'])
                df1.loc[:, 'espiral'] = i+1
                # tiempo
                df_tiempo = pd.DataFrame(vector_tiempos[i].T.astype(float), columns=['tiempos'])
                df1 = pd.concat([df1, df_tiempo], axis=1)
                # espiral todas
                espiral_pedida = espiral_pedida.append(df1, ignore_index=True)
            else:
                raise EntradaError("Error en el nombre de la espiral")

        i += 1
    if not existe:
        raise EntradaError("Error en el nombre de la espiral")
    # Devolver espiral cargada
    return espiral_pedida


def visualizar_espiral(espiral_todas, each_espiral):
    # Visualizo una espiral
    espiral_todas.loc[espiral_todas.espiral == each_espiral, :].plot('x', 'y', kind='scatter')
    plt.title("{}".format("Spiral"))
    plt.show()


def clustering(espiral_todas, metodo, file_name):
    # A continuación  los puntos exteriores no se ponen
    # en el orden correcto
    punto_central_x = 650
    punto_central_y = 350
    for index, row in espiral_todas.iterrows():
        if abs(row.x - punto_central_x) < 100 and abs(row.y - punto_central_y < 100):
            x_primer_punto = row.x
            y_primer_punto = row.y

    np.random.seed(0)

    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    n_samples = 1500
    noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)

    # ============
    # Set up cluster parameters
    # ============
    plt.figure(figsize=(20, 3), dpi=500)
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)

    plot_num = 1

    default_base = {'quantile': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 5,
                    'min_samples': 20,
                    'xi': 0.05,
                    'min_cluster_size': 0.1}

    algo_params = {'damping': .77, 'preference': -240,
                   'quantile': .2, 'n_clusters': 5,
                   'min_samples': 20, 'xi': 0.25}

    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    # Calcular centroide de todas las espirales
    dataset = espiral_todas

    x_ini = dataset.loc[:, ['x', 'y']].to_numpy()

    # normalize dataset for easier parameter selection
    scaler = StandardScaler()
    x = scaler.fit_transform(x_ini)

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(x, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(x, n_neighbors=params['n_neighbors'], include_self=False)

    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
    single = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='single', connectivity=connectivity)
    birch = cluster.Birch(n_clusters=5)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='lobpcg', affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=1, algorithm='ball_tree', min_samples=100)
    optics = cluster.OPTICS(min_samples=params['min_samples'], xi=params['xi'],
                            min_cluster_size=params['min_cluster_size'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock", n_clusters=params['n_clusters'], connectivity=connectivity)
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')

    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('Birch', birch),
        ('Single', single),
        ('SpectralClustering', spectral),
        #('DBSCAN', dbscan),
        #('Ward', ward)
    )

    for name, algorithm in clustering_algorithms:
        if metodo == name or metodo == -1:

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                            "connectivity matrix is [0-9]{1,2}" +
                            " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                            " may not work as expected.",
                    category=UserWarning)
                #
                a = scaler.inverse_transform(x, copy=None)
                algorithm.fit(a)
                print(name)
                if name == 'Birch':
                    centroids = scaler.inverse_transform(algorithm.subcluster_centers_)
                print(algorithm)
            # Creo que se podría eliminar el if
            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(int)
            else:
                y_pred = algorithm.predict(a)

            # Configuro el plot
            plt.figure()
            plt.title(f'{name}-{file_name}', size=18)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#b10dc9']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])
            plt.scatter(a[:, 0], a[:, 1], s=10, color=colors[y_pred])

            if algorithm == two_means:
                centroids = algorithm.cluster_centers_.copy()
                centro = hallar_centroide_espirales(centroids, name, x_primer_punto, y_primer_punto, file_name)
            elif algorithm == single:
                centroids = NearestCentroid()
                centroids.fit(a, y_pred)
                print(f'(Los centroides del algoritmo {name} son {centroids.centroids_})')
                centroids = centroids.centroids_
                centro = hallar_centroide_espirales(centroids, name, x_primer_punto, y_primer_punto, file_name)
                print(f'(El centro de la espiral del algoritmo {name} es {centro})')
            elif algorithm == average_linkage:
                centroids = None
            elif algorithm == dbscan:
                centroids = None
            elif algorithm == optics:
                centroids = None
            elif algorithm == birch:
                centroids = algorithm.subcluster_centers_
                centro = hallar_centroide_espirales(centroids, name, x_primer_punto, y_primer_punto, file_name)
            elif algorithm == spectral:
                plt.show()
                continue
    return centro
            # Dibujo la espiral con todas las coordenadas para ver la diferencia entre centros
            #for each_centroin in centroids:
                #coordenadas = f'({each_centroin[0].round(2)},{each_centroin[1].round(2)})'
                #plt.text(each_centroin[0], each_centroin[1], coordenadas)
                #plt.savefig(f'{name}.{file_name}.jpg')
            #plt.show()



def hallar_centroide_espirales(centroides, name, x_primer_punto, y_primer_punto, file_name):

    pandas_centroides = pd.DataFrame(centroides.astype(float), columns=['x', 'y'])
    distancia_minima = 10000
    for index, variable in pandas_centroides.iterrows():
        distancia = abs(abs(x_primer_punto - variable[0]) + abs(y_primer_punto - variable[1]))
        if distancia < distancia_minima:
            distancia_minima = distancia
            indice = index
    pandas_centroides_externos = pandas_centroides.drop(index=indice, axis=1)

    # transformo dataframe exteriores a numpy
    exteriores_numpy = pandas_centroides_externos.to_numpy()

    # Hallo el centro de las espirales a través de su media
    centro_espiral = pandas_centroides_externos.mean(axis=0).astype(float)
    # Compruebo que el centro de la espiral con el primer punto. Si hay mucha diferencia
    # significa que la espiral no ha calculado bien los centroides debido a
    # Los datos de entrada
    distancia = abs(abs(x_primer_punto - centro_espiral[0]) + abs(y_primer_punto - centro_espiral[1]))
    if distancia > 400:
        raise EntradaError("Error de los datos de entrada")

    centro_espiral_pandas = pd.DataFrame([pandas_centroides_externos.mean(axis=0)], columns=['x', 'y'])
    for i in range(0, 4, 1):
        coordenadas_externas = f'({exteriores_numpy[i][0].round(2)},{exteriores_numpy[i][1].round(2)})'
        plt.text(exteriores_numpy[i][0] - 80, exteriores_numpy[i][1], coordenadas_externas)
    coordenadas = f'({round(centro_espiral[0], 2)},{round(centro_espiral[1], 2)})'
    plt.text(centro_espiral[0], centro_espiral[1], coordenadas)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.savefig(f'resultados/{name}-{file_name}-final.jpg', dpi=200)
    plt.show()
    return centro_espiral_pandas


class EntradaError(Exception):
    pass


class ErrorDirectorio(Exception):
    pass
