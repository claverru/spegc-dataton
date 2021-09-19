# OCEAN


## Entorno

Nosotros hemos usado un contenedor de Docker como entorno de desarrollo por lo que si así se quisiera podría ejecutarse de este modo.
Es necesario tener instalado Docker, los drivers de Nvidia y nvidia-container-toolkit.
En cuanto al SO este tiene que ser Linux, Mac o WSL2 de la versión de desarrolladores de MS.

No obstante dejamos los requisitos para hacer funcionar el sistema en inferencia:

- (Opcional): CUDA 11.1 y CUDNN8
- torch==1.8.1
- torchvision==0.9.1
- albumentations
- pytorch-lightning
- timm
- git+https://github.com/openai/CLIP.git

Si se quiere usar CUDA, se recomienda instalar torch y torchvision con Conda. El resto de librerías y repositorios se pueden instalar con pip.


## Uso en inferencia

Tras descomprimir los checkpoints en la raíz del respositorio, proceder:

`python inference.py --help`

Se mostrarán todos los argumentos disponibles para la ejecución. Por ejemplo:

`python inference.py --plot-path result.png --center-crop`

Se irá guardando una figura con los resultados en el path seleccionado.
Las imágenes seran cortadas por su centro antes de realizar el resto de procesamiento.

`python inference.py --element-th 0.7`

Se aumenta el threshold de la probabilidad para considerar un elemento como predicho.

Se recomienda hacer la predicción con las imágenes en el tamaño por defecto, con el cual fueron además entrenados los modelos.

Se pueden cambiar las clases para CLIP aplicadas a los objetos de fauna de la siguiente manera (por defecto están en `src.constants.py`):

`python inference.py --fauna-classes "my first class" "my second class" ...`
