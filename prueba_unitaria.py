import unittest
import numpy
import tratamiento_espirales


class MyTestCase(unittest.TestCase):

    def test_longitud_datos(self):
        directorio = 'Data - total/'
        datos_espirales = tratamiento_espirales.cargar_directorio(directorio)
        for element in datos_espirales:
            self.assertEqual(len(element), 1)

    def test_tipo_nombre_espiral(self):
        directorio = 'Data - total/'
        datos_espirales = tratamiento_espirales.cargar_directorio(directorio)
        for element in datos_espirales:
            self.assertIs(type(element[0][0][0]), numpy.str_)

    def test_nombre_espiral_bien_formado(self):
        directorio = 'Data - total/'
        datos_espirales = tratamiento_espirales.cargar_directorio(directorio)
        for element in datos_espirales:
            self.assertIn('_spiral_', element[0][0][0])
            self.assertIn('TE', element[0][0][0])

    def test_ruta_error(self):
        directorio = 'Data- totl/'
        with self.assertRaises(tratamiento_espirales.EntradaError):
            tratamiento_espirales.cargar_directorio(directorio)

    def test_espiral_erronea(self):
        directorio = 'Data - total/'
        file_name = 'Espiral erronea'
        datos_espirales = tratamiento_espirales.cargar_directorio(directorio)
        with self.assertRaises(tratamiento_espirales.EntradaError):
            tratamiento_espirales.cargar_espiral(datos_espirales, file_name)



if __name__ == '__main__':
    unittest.main()
