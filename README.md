# proyecto_bootcamp_mintic
# Modelo predicción accidentes de tránsito Fusagasugá.
## Introducción
El proyecto aborda el análisis de los accidentes de tránsito del Municipio de Fusagasugá utilizando los datos históricos para identificar patrones y tendencias que puedan ser útiles para implementar estrategias de seguridad vial. A través del análisis de datos se pretende crear un modelo de predicción que permita a los interesados tomar decisiones, acciones y estrategias para prevenir la ocurrencia de estos. Algunos aspectos relevantes dentro del análisis son el tipo de accidente, el día, la clase de accidente y gravedad. Los resultados del análisis y el modelo como tal permitirán proponer medidas efectivas para reducir el número de accidentes y la seguridad vial del municipio de Fusagasugá.
## Objetivo General:
Desarrollar un modelo de aprendizaje supervisado utilizando regresión lineal para predecir la hora y día en la que se posiblemente se presentarán los accidentes en Fusagasugá durante el año 2024, proporcionando información para la toma de decisiones en la gestión de seguridad vial de sus habitantes como de la población flotante.
### Objetivos Específicos:
1. Identificar las principales clases de accidentes que ocurren en el Municipio, así como la frecuencia de las mismas a fin de obtener información clara respecto de las que mas afectan la accidentalidad. 
2. Mapear el impacto en la salud de las personas de la accidentalidad a partir de los datos obtenidos en la categoría Gravedad.
3. Analizar la estacional de los accidentes ocurridos en los años 2019 al 2023 con el fin de identificar tendencias y picos de acuerdo con los días de la semana en que ocurrieron de tal forma que contribuya a la implementación de medidas de prevención más eficaces.
4. Realizar la predicción de accidentes en el año 2024 identificando las horas en que ocurrirán los accidentes de tránsito.
## Delimitación del Proyecto
### Alcance: 
Este proyecto analizará los datos de accidentalidad en el municipio de Fusagasugá entre los años 2019 y 2023. El análisis se centrará en identificar los factores que influyen en la accidentalidad del municipio y definir el modelo predictivo del año 2024. 
### Fuentes de Datos: 
CSV de Datos Abiertos: “Accidentes de Tránsito registrados en el municipio de Fusagasugá”, por la Secretaría de Movilidad de la Alcaldía del municipio de Fusagasugá.
### Marco Temporal: 
2019 - 2023.
## Metodología:
Recolección y limpieza de datos: se eliminan columnas innecesarias y se renombran los campos.
EDA (Análisis de datos exploratorios): Generar estadísticas y gráficos descriptivos para identificar comportamientos de variables clave como clase de accidente y gravedad.
Identificar estacionalidad en los accidentes y determinar tendencias en la frecuencia de accidentes a lo largo del tiempo
Visualización: Uso de Python, Matplotlib y Seaborn para paneles interactivos.
## Documentación del Proyecto
Desarrollo de un modelo de aprendizaje supervisado de pronóstico utilizando regresión lineal para predecir la accidentalidad en el municipio de Fusagasugá en el año 2024.
## Instalación Librerías.
Para ello, se han importado diversas librerías de Python que permitirán el manejo, análisis y visualización de datos.
Librerías Utilizadas
pandas
matplotlib
seaborn
numpy
scikit-learn
## Paso 1: Cargue de la base de datos
Se realiza el cargue de la base de datos con la función load_file_accidentes(file_path). Esta base de datos contiene la estadística de accidentalidad de accidentes del Municipio de Fusagasugá entre Nov 2019 a junio 2024.
### Conversión Columna Fecha:
Se convierte la columna llamada "Fecha del Accidente" en tipo de dato datetime, la conversión de fecha utiliza la función errors='coerce'para identificar algún error en la columna fecha y convirtiendo valores no válidos.
## Paso 2: Análisis exploratorio de los datos
En este paso se realiza análisis exploratorio realizado sobre la base de datos de accidentes de tránsito. El propósito es realizar un análisis preliminar de los datos para comprender la estadística histórica del periodo comprendido. En este análisis se identifican patrones, se detectan anomalías y se preparan los datos para análisis más profundos. Algunos procesos que se ejecutan en este paso es identificar la estructura de los datos, identificar valores nulos y únicos para evaluar la integridad de la data.
## Limpieza de los Datos:
En el proceso de limpieza de los datos en el dataframe de accidentes de tránsito se eliminan columnas innecesarias, unifica valores en ciertas columnas, ajusta formatos de fecha y hora y renombra columnas para mayor claridad. 
### Las columnas que fueron depuradas son: 
'Informes Policiales de Accidentes de Tránsito (IPAT) ', 'Dirección', 'Barrio', 'Comuna', 'Corregimiento', 'Hipótesis',                   'Hipótesis 2', 'Motocicleta', 'Mes'] 
### Los datos que se unificaron o reemplazaron corresponden a la columna Clase_accidente y Gravedad respectivamente son: 
CAIDA OCUPANTE': 'CAIDA OCUPANTE',
        'CAÍDA': 'CAIDA OCUPANTE',
        'INCENDIO': 'INCENDIO',
        'INCENERADO': 'INCENDIO',
        'OTRO': 'OTROS',
        'OTROS': 'OTROS'
        'CON MUERTO': 'MUERTOS',
        'MUERTO': 'MUERTOS',
        'HERIDO': 'HERIDOS',
        'HERIDOS': 'HERIDOS'
Se ajusta el formato de la fecha del accidente y formato de la hora.
### Se renombran las columnas siguientes:
         'Fecha del Accidente': 'Fecha_accidente', 
        'Género': 'Genero',
        'Clase de Accidente': 'Clase_accidente',
        'Choque Con': 'Choque_con',
        'Clase de Vehículo 1': 'Clase_vehiculo_1',
        'Servicio': 'Servicio_vehiculo_1',
        'Gravedad Conductor': 'Gravedad_Conductor_vehiculo_1',
        'Embriaguez': 'Embriaguez_vehiculo_1',
        'Grado': 'Grado_vehiculo_1',
        'Clase de Vehículo 2': 'Clase_vehiculo_2',
        'Servicio 2': 'Servicio_vehiculo_2',
        'Gravedad Conductor 2': 'Gravedad_conductor_vehiculo_2',
        'Embriaguez 2': 'Embriaguez_vehiculo_2',
        'Grado 2': 'Grado_vehiculo_2'})
## Paso 3: Análisis de series temporales con descomposición:
La función analisis_series_temporales realiza un análisis de series temporales sobre el DataFrame de accidentes de tránsito. Los pasos que sigue son:
1. Verificación de la columna Fecha_accidente: Comprueba si la columna Fecha_accidente está presente en el DataFrame.
2. Creación de columnas año y mes: Extrae el año y el mes de la columna Fecha_accidente.
3. Agrupación y conteo de accidentes: Agrupa los datos por año y mes, y cuenta el número de accidentes.
4. Creación de la columna fecha: Crea una columna de fecha en formato YYYY-MM para el análisis de series temporales.
5. Establecimiento del índice: Establece la columna fecha como el índice del DataFrame.
6. Descomposición de la serie temporal: Realiza la descomposición de la serie temporal en componentes estacionales, de tendencia y residuales.
7. Visualización: Grafica los componentes de la descomposición y muestra el gráfico.

## Paso 4: Ajustes adicionales y graficas consolidados: 
Con la función ajustar y graficar se realizan ajustes adicionales en el DataFrame de accidentes de tránsito y genera una gráfica consolidada de la accidentalidad por mes y año. Los pasos que sigue son:
1. Impresión de las primeras líneas del DataFrame: Muestra las primeras filas del DataFrame depurado.
2. Conteo de accidentes por mes: Añade una columna para contar los accidentes por mes, excluyendo datos del año 2024.
3. Agrupación por mes y año: Agrupa los datos por mes y año, sumando el conteo de accidentes.
4. Generación de la gráfica: Crea una gráfica de líneas que muestra la accidentalidad consolidada por mes para los años 2019 a 2023.

## Grafico 1 y Barras ajustadas.
La función graficar_barras_agrupadas genera un gráfico de barras agrupadas que muestra la distribución de accidentes por clase y gravedad. Los pasos que sigue son:

1. Creación de columnas año: Extrae el año de la columna Fecha_accidente y filtra los datos hasta el año 2023.
2. Verificación de columnas: Comprueba si las columnas Clase_accidente y Gravedad están presentes en el DataFrame.
3. Creación de tabla de contingencia: Cuenta las ocurrencias de accidentes según la clase y la gravedad.
4. Generación del gráfico: Utiliza Seaborn para crear un gráfico de barras agrupadas que muestra la distribución de accidentes por clase y gravedad.
5. Configuración del gráfico: Ajusta el título, las etiquetas y la leyenda del gráfico, y muestra las etiquetas de los valores en cada barra.


   ![Imagen de WhatsApp 2024-10-25 a las 19 29 51_32c7c20e](https://github.com/user-attachments/assets/f4d9f776-b34f-4c62-bbfa-c2bbe2c39e44)


## Paso 5: Creación código de Análisis estacional de los accidentes:
### Creación de un Diccionario para la Traducción de los Nombres
Se ha creado un diccionario para traducir los nombres de los días de la semana del inglés al español.
dias_semana_espanol = {
    'Monday': 'Lunes',
    'Tuesday': 'Martes',
    'Wednesday': 'Miércoles',
    'Thursday': 'Jueves',
    'Friday': 'Viernes',
    'Saturday': 'Sábado',
    'Sunday': 'Domingo'
### Creación de Columna con Día de la Semana
Se añadió una columna al DataFrame con el día de la semana correspondiente a la fecha del accidente.
df_depurado['Dia_semana'] = df_depurado['Fecha_accidente'].dt.day_name().map(dias_semana_espanol)

### Agrupación Mensual y Diaria de Accidentes
Se agruparon los datos por mes y año, así como por días de la semana.
df_mensual = df_depurado.groupby(['año', 'mes']).agg({'conteo_accidentes': 'sum'}).reset_index()
df_dia = df_depurado.groupby(['Dia_semana']).agg({'conteo_accidentes': 'sum'}).reset_index()

## Visualización de los Datos:

Se generaron gráficos para analizar la estacionalidad de los accidentes:

### Gráfica consolidada por año
plt.figure(figsize=(10,6))
sns.lineplot(df_mensual, x = 'mes', y = 'conteo_accidentes', hue = 'año', palette='bright')
plt.title('Accidentalidad consolidada 2019 - 2023', fontsize=16)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Accidentalidad', fontsize=12)
plt.xticks(range(1,13), ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'])
plt.legend(title='año')
plt.show()

![Imagen de WhatsApp 2024-10-25 a las 18 36 16_6480506a](https://github.com/user-attachments/assets/e68a6614-b908-4fc6-9be8-a66c7ad1b9c3)


### Gráfica de estacionalidad por días de la semana

plt.figure(figsize=(10,6))
sns.lineplot(df_dia, x = 'Dia_semana', y = 'conteo_accidentes')
plt.title('Estacionalidad de los accidentes en Fusagasugá del 2019 al 2023', fontsize=16)
plt.xlabel('Día de la semana', fontsize=12)
plt.ylabel('Accidentalidad', fontsize=12)
plt.xticks(range(0,7), ['Lun','Mar','Mie','Jue','Vie','Sab','Dom'])
plt.show()

![Imagen de WhatsApp 2024-10-25 a las 18 38 11_950017a8](https://github.com/user-attachments/assets/ac5d635d-98d9-4309-8414-b30f69c49935)

## Paso 6: Creación Código para la Predicción de Horas de Accidentes en 2024
### Simulación de Datos
Se simulan los datos de horas de accidentalidad y los conteos de accidentes.
X = df_depurado['Hora'].values.reshape(-1, 1)  # Horas de la accidentalidad
y = df_depurado['conteo_accidentes'].values  # Conteo de accidentes

### División del Conjunto de Datos
El conjunto de datos se divide en entrenamiento (80%) y prueba (20%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Creación y Entrenamiento del Modelo de Regresión Lineal
Se crea y entrena un modelo de regresión lineal con los datos de entrenamiento.
model = LinearRegression()
model.fit(X_train, y_train)

### Realización de Predicciones
Se realizan predicciones en el conjunto de prueba.
y_pred = model.predict(X_test)

### Evaluación del Modelo
Se evalúa el modelo utilizando métricas como el error cuadrático medio (MSE) y el coeficiente de determinación (R²).
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Coeficiente de regresión (pendiente): {model.coef_[0]}")
print(f"Término independiente (intercepto): {model.intercept_}")
print(f"Error cuadrático medio (MSE): {mse}")
print(f"Coeficiente de determinación (R²): {r2}")

### Visualización de Resultados
Se visualizan los resultados con una gráfica de dispersión y la línea de regresión.
plt.scatter(X, y, color='green', label="Datos reales")
plt.plot(X_test, y_pred, color='red', label="Línea de regresión")
plt.title('Regresión Lineal Simple: Horas ocurrencia accidentes vs. Accidentalidad')
plt.xlabel('Horas ocurrencia accidentes')
plt.ylabel('Accidentalidad')
plt.legend()
plt.show()

### Visualización de las Primeras Líneas del DataFrame Depurado
Se muestran las primeras líneas del DataFrame para confirmar la correcta ejecución del preprocesamiento.
print(df_depurado.conteo_accidentes.dtype)
print(df_depurado.conteo_accidentes)

![Imagen de WhatsApp 2024-10-25 a las 18 49 50_5c05b638](https://github.com/user-attachments/assets/531b4d68-6f57-4496-89dc-d130e549b218)

## Información componentes del CSV

Información de la entidad:

Departamento: Cundinamarca
Municipio Fusagasugá
Nombre de la Entidad: Alcaldía de Fusagasugá
Área o dependencia:Secretaría de Movilidad


Información de Datos

Idioma:Español
Cobertura Geográfica: Municipal
Frecuencia de Actualización: Anual
Fecha Emisión (aaaa-mm-dd): 2024-05-3

Filas: 1.392
Columnas: 27

Columnas: Informes Policiales de Accidentes de Tránsito (IPAT), Fecha de Accidente, Dirección, Latitud, Longitud, Barrio, Comuna, Corregimiento, Genero, Hora, Gravedad, Clase de Accidente, Choque Con, Clase de Vehículo 1, Servicio, Gravedad Conductor, Embriaguez, Grado, Clase de Vehículo 2, Servicio 2,Hipótesis 2,Gravedad Conductor 2, Embriaguez 2, Grado 2, Motocicleta, Mes

## Conclusiones:

1. La alta accidentalidad se prresenta durante los fines de semanaa en las horas de la noche y en la madrugada. Se deben aunar esfuerzos por parte de autoridades y organismos de prevención municipal y Departamental para adoptar medidas de control para prevenir accidentes especialmente en estas horas y días mencionados. El numero de accidentes podría reducir en un 25% solo si se toman acciones durante el fin de semana. 

## Referencias:

Ministerio de Transporte de Colombia. (n.d.). Accidentes de Tránsito registrados en el municipio. Datos Abiertos Colombia. Recuperado de https://www.datos.gov.co/Transporte/Accidentes-de-Tr-nsito-registrados-en-el-municipio/22ip-4jk2/about_data
