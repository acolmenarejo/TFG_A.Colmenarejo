import csv
import sys
from scipy import stats
from sklearn.linear_model import LinearRegression
import re
from tratamiento_espirales import *
from analisis_tiempo import cargar_tiempo
from informe_final import analisis
from sklearn import metrics
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


# file_nam se recibe desde interfaz.py
def main(modalidad, ruta, file_name, visualizacion):
    # Eliminar contenido de csv si hubiera
    TEXTFILE = open("centro.csv", "w")
    TEXTFILE.truncate()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)
    # Crear directorios para almacenar resultados
    dirName = './resultados'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directorio resultados", dirName, " Creado ")

    dirName2 = './Informe'
    if not os.path.exists(dirName2):
        os.mkdir(dirName2)
        print("Directorio Informe", dirName2, " Creado ")

    # todos_archivos contiene todos los datos de las espirales
    todos_archivos = cargar_directorio(ruta)
    lado_original = []

    text = file_name
    # Espiral tratada o sin tratar obtengo las cadenas inicial y final para buscar en el directorio coincidencias
    cadena_final = re.search(r'_spiral_(.*?)$', text).group(1)
    cadena_inicial = re.search(r'(.*?)_spiral', text).group(1)
    # Divido la cadena inicial en dos
    numero_paciente, numero_de_prueba = cadena_inicial.split('_')

    # Nombres de todas las espirales
    nombres_en_directorio = get_nombres()
    # Cargar espirales con nombres similares. Modalidad 1: Treated vs not Treated
    # Modalidad 1: Cargo dos espirales tratada y sin tratar y las analizo separandamente
    for i in nombres_en_directorio:
        cadena_final1 = re.search(r'_spiral_(.*?)$', i).group(1)
        cadena_inicial1 = re.search(r'(.*?)_spiral', i).group(1)
        numero_paciente1, numero_de_prueba1 = cadena_inicial1.split('_')
        if modalidad == 1:
            if cadena_final != cadena_final1 and cadena_inicial == cadena_inicial1:
                new_file = cadena_inicial1 + '_spiral_' + cadena_final1
                break
        else:  # Modalidad 2. Todas las tratadas y todas las sin tratar
            # Primero cojo las tres de un lado
            if cadena_final == cadena_final1 and numero_paciente1 == numero_paciente:
                new_file = cadena_inicial1 + '_spiral_' + cadena_final1
                lado_original.append(new_file)
    # Si en este punto newfile está vacío, ERROR
    try:
        # Puede que new_file no exista y dé error
        new_file
    except NameError:
        print('ERROR - Faltan más espirales del paciente')
        sys.exit()
    if modalidad == 1:
        espiral1 = clustering_espirales(todos_archivos, file_name)
        espiral2 = clustering_espirales(todos_archivos, new_file)
        espiral_a_tratar = pd.concat([espiral1, espiral2])
    elif modalidad == 2:
        espiral_a_tratar = pd.DataFrame()
        for espiral in lado_original:
            # Clustering de cada espiral
            appended_espirals = clustering_espirales(todos_archivos, espiral)
            if not espiral_a_tratar.empty:
                espiral_a_tratar = pd.concat([espiral_a_tratar, appended_espirals])
            else:
                espiral_a_tratar = appended_espirals
    i = 1
    count = 0
    contador_espirales = 0
    color = ['tab:blue', 'tab:orange', 'tab:green']
    espiral_a_juntar = pd.DataFrame()
    # Obtengo los nombres de las espirales
    if modalidad == 2:
        nombres_espirales = lado_original
    else:
        nombres_espirales = list(file_name.split(" "))
        nombres_espirales.append(new_file)
    numero_espiral = 0

    for each_espiral in espiral_a_tratar.espiral.unique():
        linea = 1
        count += 1
        # Tomo una a una todas las espirales o la espiral concreta
        espiral = espiral_a_tratar.loc[espiral_a_tratar.espiral == each_espiral, :]
        # Visualizo la espiral "ESPIRAL"
        visualizar_espiral(espiral, each_espiral)
        # Cojo una fila del fichero centro csv
        i = 2 * count
        with open("centro.csv", "r") as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                if linea == i:
                    centro = row
                    # print(centro)
                    break
                linea += 1

        # Resto el centro de esa espiral
        puntos_todos = resto_centro_espiral(espiral, centro)
        # Pasar a polares
        puntos_polares = paso_a_polares(puntos_todos)
        # Eliminos los puntos externos
        puntos_espiral = elimino_puntos_externos(puntos_polares)
        # El límite en grados es 180 y lo paso a 360
        espiral_grados = sumar_vueltas(puntos_espiral)
        print('Espiral transformada a polares')
        # Guardo las espirales en un dataframe
        if not espiral_a_juntar.empty:
            espiral_a_juntar = pd.concat([espiral_a_juntar, espiral_grados])
        else:
            espiral_a_juntar = espiral_grados

        if visualizacion == 'separadas':
            visualizo_separadas(espiral_grados, nombres_espirales, file_name, visualizacion, modalidad, numero_espiral)
        numero_espiral += 1
    if visualizacion == 'separadas':
        print('Fin del programa')
        sys.exit()

    # Espirales juntas
    if visualizacion == 'juntas':
        plt.cla()
        plt.clf()
        # Representacion de 3 espirales en polares
        for each_espiral in espiral_a_juntar.espiral.unique():
            each_espiral.astype(int)
            legend = f'Nombre espiral: {str(nombres_espirales[contador_espirales])}'
            # Tomo una a una todas las espirales o la espiral concreta
            espiral_juntas = espiral_a_juntar.loc[espiral_a_juntar.espiral == each_espiral, :]
            plt.plot(espiral_juntas['ang_grados'], espiral_juntas['r'], c=color[contador_espirales], label=legend)
            contador_espirales += 1
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.title("Grados - Radio", size=20)
        plt.legend(loc="upper left")
        plt.savefig(f'resultados/Grados-Radio' + file_name + '-JUNTAS.jpg', dpi=200)
        plt.show()
        plt.cla()
        plt.clf()
        contador_espirales = 0
        fig, axs = plt.subplots(1, 2, tight_layout=True)
        # Representacion 3 espirales
        for each_espiral in espiral_a_juntar.espiral.unique():
            each_espiral.astype(int)
            # Tomo una a una todas las espirales o la espiral concreta
            espiral_juntas = espiral_a_juntar.loc[espiral_a_juntar.espiral == each_espiral, :]

            # Calculo regresion
            grados = espiral_juntas['ang_grados'].values.reshape(-1, 1)
            modulo = espiral_juntas['r'].values.reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(grados, modulo)
            pred_modulo = lr.predict(grados)

            # Subplots
            axs[0].plot(grados, modulo - pred_modulo, color=color[contador_espirales])
            axs[1].hist(modulo - pred_modulo, bins=600, orientation="horizontal", color=color[contador_espirales])

            contador_espirales += 1
        axs[0].xaxis.set_tick_params(labelsize=14)
        axs[0].yaxis.set_tick_params(labelsize=14)
        axs[1].xaxis.set_tick_params(labelsize=14)
        axs[1].yaxis.set_tick_params(labelsize=14)
        axs[1].set_xlim([0, 15])
        axs[1].set_ylim([-90, 90])
        axs[1].set_title('Histograma del error', fontsize=12)
        axs[0].set_title('Error respecto a la regresión', fontsize=12)

        plt.savefig(f'resultados/Error' + file_name + '-JUNTAS.jpg', dpi=200)
        plt.show()
        plt.cla()
        plt.clf()
        contador_espirales = 0
        # Representacion tiempo
        for each_espiral in espiral_a_juntar.espiral.unique():
            each_espiral.astype(int)
            # Tomo una a una todas las espirales o la espiral concreta
            espiral_juntas = espiral_a_juntar.loc[espiral_a_juntar.espiral == each_espiral, :]
            # Analisis tiempo
            espiral_juntas.loc[:, 'tiempos'] = espiral_juntas.loc[0:, 'tiempos'] - espiral_juntas.loc[0, 'tiempos']
            espiral_juntas.loc[:, 'diff'] = espiral_juntas.loc[:, 'r'].diff()
            for index, row in espiral_juntas.iterrows():
                # Para evitar pintar cuando deje el lápiz pausado
                if abs(espiral_juntas.loc[index, 'diff']) < 0.01 or abs(espiral_juntas.loc[index, 'diff']) > 100:
                    espiral_juntas = espiral_juntas.drop(index=index, axis=1)
            plt.plot(espiral_juntas['tiempos'], espiral_juntas['diff'], ',-', color=color[contador_espirales])
            contador_espirales += 1
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.title("Variación del radio vs tiempo", size=20)
        plt.savefig(f'resultados/Tiempo' + file_name + '-JUNTAS.jpg', dpi=200)
        plt.show()

    # Análisis de resultados

    analisis(file_name, visualizacion, modalidad)


# A partir de aquí todas las funciones

# Clustering de la espiral, edita centro.csv y devuelve la espiral tras comprobar que es correcta
def clustering_espirales(todos_archivos, file_name):
    espiral_a_tratar = cargar_espiral(todos_archivos, file_name)
    # Calculo el centro de la espiral
    print("Calculando el centro de la espiral a través de clustering")

    # -1 para todos los métodos, escribir nombre para uno concreto
    centro = clustering(espiral_a_tratar, 'Single', file_name)  # 'Birch')
    centro.to_csv('centro.csv', index=False, mode='a')
    return espiral_a_tratar


def resto_centro_espiral(espiral, centro):
    puntos_todos = pd.DataFrame(espiral.astype(float), columns=['x', 'y', 'espiral', 'tiempos', 'x_', 'y_'])
    # AHORA RESTAR EL CENTROIDE HALLADO en clustering
    x_centro = float(centro[0])
    y_centro = float(centro[1])

    # Copio columna x en x_
    puntos_todos['x_'] = puntos_todos['x']
    puntos_todos['y_'] = puntos_todos['y']

    puntos_todos['x_'] = puntos_todos['x_'] - x_centro
    puntos_todos['y_'] = puntos_todos['y_'] - y_centro

    plt.title("Espiral con centro (0,0)")
    plt.plot(puntos_todos['x_'], puntos_todos['y_'], '.-')
    plt.show()

    return puntos_todos


def paso_a_polares(puntos_todos):
    # Pasamos a polares
    puntos_todos['r'] = np.sqrt(puntos_todos['x_'] ** 2 + puntos_todos['y_'] ** 2)
    puntos_todos['ang_grados'] = (np.degrees(np.arctan2(puntos_todos['x_'], puntos_todos['y_'])) - 90) % 360
    puntos_todos['t_polar'] = np.arctan2(puntos_todos['x_'], puntos_todos['y_'])
    return puntos_todos


def elimino_puntos_externos(puntos_todos):
    # Usaré método 1
    # Métodos1: Puntos alejados más de "R" del centro
    # puntos_espiral =  puntos_todos sin los cuatro puntos
    puntos_espiral = puntos_todos[(puntos_todos['r'] < 450)]
    plt.scatter(puntos_espiral['x_'], puntos_espiral['y_'])
    plt.title("Eliminación puntos externos con el radio")
    plt.show()

    # Métodos2: Puntos con menor desviación típica, respecto la media
    puntos_todos['z_score'] = stats.zscore(puntos_todos['r'])
    puntos_espiral_z = puntos_todos[puntos_todos['z_score'] < 0]
    plt.scatter(puntos_espiral_z['x_'], puntos_espiral_z['y_'])
    plt.title("Eliminación puntos externos con Z-Score")
    plt.show()
    return puntos_espiral


def sumar_vueltas(puntos):
    count = 0
    puntos = puntos.reset_index()
    radio_maximo = puntos.loc[:, 'r'].max()
    indexlast = puntos.index[-1]
    puntos.loc[:, 'dif'] = puntos.loc[:, 'ang_grados'].diff()
    puntos.loc[:, 'vuelta'] = 0

    for index, row in puntos.iterrows():
        if row["r"] < 0.05 * radio_maximo and abs(row["dif"]) > 100:
            puntos.loc[:index, 'ang_grados'] += 180
            puntos.loc[:, 'dif'] = puntos.loc[:, 'ang_grados'].diff()
        if row["r"] < 0.05 * radio_maximo:
            puntos = puntos.drop(count, axis=0)
        if row["dif"] < -300:
            puntos.loc[index:indexlast, 'vuelta'] += 1
            puntos.loc[index:indexlast, 'ang_grados'] += 360
            puntos.loc[:, 'dif'] = puntos.loc[:, 'ang_grados'].diff()
        elif row["dif"] > 300:
            puntos.loc[index:indexlast, 'vuelta'] -= 1
            puntos.loc[index:indexlast, 'ang_grados'] -= 360
            puntos.loc[:, 'dif'] = puntos.loc[:, 'ang_grados'].diff()

        count += 1

    puntos['moving_average'] = puntos['ang_grados'].ewm(alpha=0.95, adjust=False).mean()
    return puntos


def visualizo_polares(normalized_puntos):
    plt.scatter(normalized_puntos['ang_grados'], normalized_puntos['r'])
    plt.title("Grados - Radio")
    plt.savefig(f'resultados/Grados - Radio.jpg', dpi=200)
    plt.show()


def calcular_error(espiral, nombres_espirales, numero_espiral):
    # Calculo regresion
    plt.cla()
    plt.clf()
    grados = espiral['ang_grados'].values.reshape(-1, 1)
    modulo = espiral['r'].values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(grados, modulo)
    pred_modulo = lr.predict(grados)
    # Evaluo la calidad de la prediccion
    score = lr.score(grados, pred_modulo)
    'Prediction score ={score}'
    # Dibujo la regresión lineal
    plt.plot(grados, pred_modulo, color='red')
    text = 'Mean Absolute Error:' + str(metrics.mean_absolute_error(modulo, pred_modulo).round(2)) +'\n'+'Mean Squared Error:' + str(metrics.mean_squared_error(modulo, pred_modulo).round(2)) +'\n'+ 'Root Mean Squared Error:' + str(
        np.sqrt(metrics.mean_squared_error(modulo, pred_modulo).round(2))) +'\n'+ 'Prediction score:' + str(score)
    plt.figtext(0.05,0.00, text, fontsize=12, va="top", ha="left")
    plt.xticks(size=18)
    plt.yticks(size=18)
    plt.plot(espiral['ang_grados'], espiral['r'], color='mediumblue')
    plt.title("Regresión: Grados - Radio", fontsize=18)
    plt.savefig(f'resultados/Regresion-{nombres_espirales[numero_espiral]}.jpg', dpi=200, bbox_inches = "tight")
    plt.show()
    # Subplots
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    axs[0].plot(grados, modulo - pred_modulo, color='indianred')
    axs[0].set_title('Error respecto a la regresión', fontsize=12)
    axs[1].hist(modulo - pred_modulo, bins=600, orientation="horizontal", color='red')
    axs[0].xaxis.set_tick_params(labelsize=14)
    axs[0].yaxis.set_tick_params(labelsize=14)
    axs[1].xaxis.set_tick_params(labelsize=14)
    axs[1].yaxis.set_tick_params(labelsize=14)
    axs[1].set_xlim([0, 15])
    axs[1].set_ylim([-90, 90])
    axs[1].set_title('Histograma del error', fontsize=14)
    plt.savefig(f'resultados/Error-{nombres_espirales[numero_espiral]}.jpg', dpi=200)
    plt.show()
    plt.cla()
    plt.clf()


def visualizo_separadas(espiral_grados, nombres_espirales, file_name, visualizacion, modalidad, numero_espiral):
    # Visualizo la espiral en polares
    visualizo_polares(espiral_grados)
    # Visualizo regresión e histograma
    calcular_error(espiral_grados, nombres_espirales, numero_espiral)
    # Represento respecto al tiempo
    cargar_tiempo(espiral_grados, nombres_espirales[numero_espiral])  # Antes puntos_espiral
    # Análisis
    analisis(nombres_espirales[numero_espiral], visualizacion, modalidad)


# main(2, 'Data - total/', 'TE02_02_spiral_treated', 'juntas')

