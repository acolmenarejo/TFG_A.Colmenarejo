import matplotlib.pyplot as plt
import pandas as pd


def cargar_tiempo(tiempos, nombre_espiral):
    pd.options.mode.chained_assignment = None  # default='warn'
    # Analisis tiempo
    tiempos.loc[:, 'tiempos'] = tiempos.loc[0:, 'tiempos'] - tiempos.loc[0, 'tiempos']
    tiempos.loc[:, 'diff'] = tiempos.loc[:, 'r'].diff()
    for index, row in tiempos.iterrows():
        if abs(tiempos.loc[index, 'diff']) < 0.01 or abs(tiempos.loc[index, 'diff']) > 100:
            tiempos = tiempos.drop(index=index, axis=1)
    plt.gca()
    plt.clf()
    plt.plot(tiempos['tiempos'], tiempos['diff'], ',-', color='turquoise')
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title("Variación del radio vs tiempo", size=20)
    plt.savefig(f'resultados/Tiempo' + nombre_espiral + '.jpg', dpi=200)
    plt.show()
