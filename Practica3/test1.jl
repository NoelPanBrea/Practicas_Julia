# Archivo de pruebas para realizar autoevaluación de algunas funciones de los ejercicios

# Importamos el archivo con las soluciones a los ejercicios
include("soluciones.jl");
#   Cambiar "soluciones.jl" por el nombre del archivo que contenga las funciones desarrolladas



# Fichero de pruebas realizado con la versión 1.11.2 de Julia
println(VERSION)
#  y la 1.11.3 de Random
using Random; println(Random.VERSION)
#  y la versión 0.14.25 de Flux
import Pkg
Pkg.status("Flux")

# Es posible que con otras versiones los resultados sean distintos, estando las funciones bien, sobre todo en la funciones que implican alguna componente aleatoria

# Para la correcta ejecución de este archivo, los datasets estarán en las siguientes carpetas:
datasetFolder = "../datasets"; # Incluye el dataset MNIST
imageFolder = "../datasets/images";
# Cambiadlas por las carpetas donde tengáis los datasets y las imágenes

@assert(isdir(datasetFolder))
@assert(isdir(imageFolder))

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------


imageFileNames = fileNamesFolder(imageFolder,"tif");
@assert(imageFileNames == ["cameraman", "lake", "lena_gray_512", "livingroom", "mandril_gray", "peppers_gray", "pirate", "walkbridge"]);

inputs, targets = loadDataset("sonar", datasetFolder; datasetType=Float32);
@assert(size(inputs)==(208,60))
@assert(length(targets)==208)
@assert(eltype(inputs)==Float32)
@assert(eltype(targets)==Bool)

image = loadImage("cameraman", imageFolder; datasetType=Float64, resolution=64)
@assert(size(image)==(64,64))
@assert(eltype(image)==Float64)

imagesNCHW = loadImagesNCHW(imageFolder; datasetType=Float64, resolution=32)
@assert(size(imagesNCHW)==(8,1,32,32))
@assert(eltype(imagesNCHW)==Float64)


MNISTDataset = loadMNISTDataset(datasetFolder; labels=[3,6,9], datasetType=Float64)
@assert(size(MNISTDataset[1])==(17998, 1, 28, 28))
@assert(eltype(MNISTDataset[1])==Float64)
@assert(length(MNISTDataset[2])==17998)
@assert(sort(unique(MNISTDataset[2]))==[3,6,9])
@assert(size(MNISTDataset[3])==(2977, 1, 28, 28))
@assert(eltype(MNISTDataset[3])==Float64)
@assert(length(MNISTDataset[4])==2977)
@assert(sort(unique(MNISTDataset[4]))==[3,6,9])


MNISTDataset = loadMNISTDataset(datasetFolder; labels=[2,7,-1], datasetType=Float32)
@assert(size(MNISTDataset[1])==(60000, 1, 28, 28))
@assert(eltype(MNISTDataset[1])==Float32)
@assert(length(MNISTDataset[2])==60000)
@assert(eltype(MNISTDataset[2])<:Integer)
@assert(sort(unique(MNISTDataset[2]))==[-1,2,7])
@assert(size(MNISTDataset[3])==(10000, 1, 28, 28))
@assert(eltype(MNISTDataset[3])==Float32)
@assert(length(MNISTDataset[4])==10000)
@assert(eltype(MNISTDataset[4])<:Integer)
@assert(sort(unique(MNISTDataset[4]))==[-1,2,7])


sinEncoding, cosEncoding = cyclicalEncoding([1, 2, 3, 2, 1, 0, -1, -2, -3]);
@assert(all(isapprox.(sinEncoding, [-0.433883739117558, -0.9749279121818236, -0.7818314824680299, -0.9749279121818236, -0.433883739117558, 0.43388373911755823, 0.9749279121818236, 0.7818314824680298, 0.0]; rtol=1e-4)))
@assert(all(isapprox.(cosEncoding, [-0.9009688679024191, -0.2225209339563146, 0.6234898018587334, -0.2225209339563146, -0.9009688679024191, -0.900968867902419, -0.22252093395631434, 0.6234898018587336, 1.0]; rtol=1e-4)))


inputs, targets = loadStreamLearningDataset(datasetFolder; datasetType=Float64)
@assert(size(inputs)==(45312,7))
@assert(length(targets)==45312)
@assert(eltype(inputs)==Float64)
@assert(eltype(targets)==Bool)



