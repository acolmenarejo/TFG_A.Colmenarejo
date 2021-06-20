from time import sleep
import sys
from _main import main


if __name__ == '__main__':

    while True:
        print('Elige el proceso...\n')
        opciones = [dict(valor='1', descripcion='Informe de Tratado vs No tratado'),
                    dict(valor='2', descripcion='Informe de un hemicuerpo'),
                    dict(valor='3', descripcion='Informe individual de espirales'),
                    dict(valor='4', descripcion='Salir')
                    ]

        for each in opciones:
            print(f"{each['valor']:>5s}.- {each['descripcion']}")
        proceso_a_ejecutar = input()

        seleccionado = None
        for each in opciones:
            if proceso_a_ejecutar == each['valor']:
                seleccionado = each
                break
        if seleccionado is None:
            print('Opción no válida\n')
            sleep(2)  # pausar 2 seg
            print(chr(27) + '[2J')
        elif seleccionado['valor'] == '4':
            sys.exit()
        else:
            print(f"Ejecutando... {seleccionado['valor']}.- {seleccionado['descripcion']}")
            print(f'Introduce el directorio donde se encuentran las espirales')
            ruta = input()
            # Introuducir el nombre del archivo
            print(f'Introduce el nombre de la espiral a diagnosticar')
            file_name = input()
            if seleccionado['valor'] in ['0']:
                pass
            elif proceso_a_ejecutar == '1':

                main(1, ruta, file_name, "juntas")
            elif proceso_a_ejecutar == '2':

                main(2, ruta, file_name, "juntas")
            elif proceso_a_ejecutar == '3':
                print(f'Introduce la forma de visualización del informe: (1) para Treated vs No_Treated y (2) para hemicuerpo')
                modalidad = input()
                main(int(modalidad), ruta, file_name, "separadas")
