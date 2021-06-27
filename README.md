# TFG_A.Colmenarejo

MANUAL DE EJECUCIÓN DE LA APLICACIÓN

Para la realización de los siguientes pasos es imprescindible tener descargado un Docker.
Pasos a realizar para la correcta ejecución del programa:

IMPORTANTE: Aunque haya un menú, se debe escribir la opción 4 salir y volver a ejecutar tras cada iteracción. NO EJECUTAR DOS COMANDOS OPCIONES DEL MENÚ SIN ANTES SALIR.


- PASOS:
Abrimos una terminal CMD como administrador y nos situamos en el directorio donde está el proyecto.

1. Comprobar si existen contenedores o imágenes previas:

 		docker ps -a

  		docker images

2. Eliminar (si existen) todos los contenedores e imágenes previas:

  		docker stop {CONTAINER ID}

		docker rm {CONTAINER ID}

 		docker image rm {IMAGE ID}
  
  (Container id e image id se obtienen del paso 1)
  
3. Crear imagen:

  		docker build -t nombreimagen:version . (Añadir ./ en vez de . al final si hubiera problemas con el path)

 	Ejemplo: docker build -t tembloresencial:4 .
  
4. Crear el contenedor:

  		docker run -ti nombreimagen:version

  	Ejemplo: docker run -ti tembloresencial:4
  
5. Interactúe con el menú de la aplicación para elaborar los informes que considere

6. Comprobar container names:

 	docker ps -a
  
7. Copiar los informes del contenedor en nuestro directorio:

  		docker cp {container name}:/TFG/Informe nombre_carpeta_destino
  	Ejemplo: docker cp cool_ritchie:/TFG/Informe informe_docker
  
  (OPCIONAL): Copiar también la carpeta resultados para ver todas las imágenes individuales generadas:

	docker cp {container name}:/TFG/resultados nombre_carpeta_destino

   Ejemplo: docker cp cool_ritchie:/TFG/resultados resultados_docker
    
8. Acceder a la nueva carpeta Informe creada en el directorio actual para ver los informes
 
9. Para ejecutar otra vez la aplicación, ejecutar los pasos 4-8 ambos incluidos.

(OPCIONAL): Se puede eliminar imágenes y contenedores como en el paso 2 antes de ejecutar el paso 3 o 4 para una mayor claridad.

Se puede cambiar el contenido del dockerfile (se puede abrir con la app bloc de notas) para que ejecute la prueba unitaria:

   Cambiar CMD  python3 interfaz.py por CMD  python3 prueba_unitaria.py y guardar. Ejecutar los pasos anteriores desde el principio.
