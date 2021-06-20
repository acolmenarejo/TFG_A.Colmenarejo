import matplotlib.pyplot as plt
import matplotlib.image as mpimg

count = 0


# Cada gráfico a visualizar entra uno por uno al método análisis hasta que hayan entrado 6
def analisis(nombre_espiral, visualizacion, modalidad):
    nombre = str(nombre_espiral)
    if visualizacion == 'juntas':
        img1 = mpimg.imread('resultados/Single-' + nombre + '-final.jpg')
        img2 = mpimg.imread('resultados/Tiempo' + nombre + '-JUNTAS.jpg')
        img3 = mpimg.imread('resultados/Grados-Radio' + nombre + '-JUNTAS.jpg')
        img4 = mpimg.imread('resultados/Error' + nombre + '-JUNTAS.jpg')

        fig = plt.figure(tight_layout=True)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(img2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(img3)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img4)

        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')

        fig.suptitle('Análisis: ' + nombre + '(' + visualizacion + ')' + '-Modalidad:' + str(modalidad), size=16)
        plt.savefig('Informe/Analisis- ' + nombre + '-' + visualizacion + '-' + 'Modalidad-' + str(modalidad) + '.jpg', dpi=200)
        plt.show()
    else:
        img1 = mpimg.imread('resultados/Single-' + nombre + '-final.jpg')
        img2 = mpimg.imread('resultados/Tiempo' + nombre + '.jpg')
        img3 = mpimg.imread('resultados/Regresion-' + nombre + '.jpg')
        img4 = mpimg.imread('resultados/Error-' + nombre + '.jpg')

        fig = plt.figure(tight_layout=True)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(img2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(img3)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img4)

        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')

        fig.suptitle('Análisis: ' + nombre + '(' + visualizacion + ')' + '-Modalidad:' + str(modalidad), size=12)
        plt.savefig('Informe/Analisis- ' + nombre + '-' + visualizacion + '-' + 'Modalidad-' + str(modalidad) + '.jpg', dpi=200)
        plt.show()
    print(f'Informe {nombre} generado. Compruebe la carpeta /Informe')


