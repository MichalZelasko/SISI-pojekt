from lib2to3.pytree import convert
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

width = 541
heigth = 555

def convertPIL(filename = "test.png") :
    im = Image.open(filename)
    
    rgb_im = im.convert('LA')
    rgb_im.show()

def readImage(filename = "test.png") :
    image = tf.keras.utils.load_img(filename)
    im = Image.open(filename)
    width = im.height
    heigth = im.width
    print(width, heigth)
    picture = np.array(image).reshape((width, heigth, 3)) / 255
    drawImage(image, shape = (width, heigth), rgb = True, scale = 1)
    return picture

def drawImage(picture, shape = None, rgb = False, scale = 255) :
    if rgb : image = Image.fromarray(scale * np.array(picture).reshape((shape[0], shape[1], 3))) 
    else : image = Image.fromarray(scale * np.array(picture).reshape((shape[0], shape[1]))) 
    image.show()
    # print('Do continue: ')
    # while input() not in ['c', 'C', 'k', 'K', 't', 'T', 'y', 'Y'] :
    #     print('Do continue: ')
    return image

def grayscaleKernel(shape, dtype = np.float64) :
    kernel = np.array([0.15, 0.7, 0.15]).reshape(shape)
    return kernel

def toGrey(picture) :
    input_shape = (1, width, heigth, 3)
    picture = np.array(picture.reshape(input_shape), dtype = np.float64)
    print(picture)
    converted = tf.keras.layers.Conv2D(filters = 1, strides = 1, kernel_size = (1, 1), padding = 'same', kernel_initializer = grayscaleKernel, input_shape=input_shape[1:])(picture)
    return converted

def pool(picture) :
    picture = tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(picture)
    return picture

def normalizedWindowKernel(shape, dtype = np.float64) :
    n = (shape[0] - 1) / 2.0
    kernel = np.ndarray((shape[0], shape[0]))
    s = 0
    for i in range(shape[0]) :
        for j in range(shape[1]) :
            dis = np.sqrt(np.power((i - n), 2.) + np.power((j - n), 2.))
            kernel[i][j] = np.exp(-np.power(dis, 2.) / (2 * np.power(n, 2.)))
            s += kernel[i][j]
    kernel = kernel / s
    return kernel.reshape(shape)

def gaussianBlur(picture) :
    input_shape = np.shape(picture)
    converted = tf.keras.layers.Conv2D(filters = 1, strides = 1, kernel_size = (3, 3), padding = 'same', kernel_initializer = normalizedWindowKernel, input_shape=input_shape[1:])(picture)
    return converted

def sobelFilterKernel1(shape, dtype = np.float64) :
    return np.array([[1, 2, 1], 
                     [0, 0, 0],
                     [-1, -2, -1]]).reshape(shape)

def sobelFilterKernel2(shape, dtype = np.float64) :
    return sobelFilterKernel1(shape, dtype = dtype).transpose().reshape(shape)

def sobelFilter(picture) :
    input_shape = np.shape(picture)
    channel1 = tf.keras.layers.Conv2D(filters = 1, strides = 1, kernel_size = (3, 3), padding = 'same', kernel_initializer = sobelFilterKernel1, input_shape=input_shape[1:])(picture)
    channel2 = tf.keras.layers.Conv2D(filters = 1, strides = 1, kernel_size = (3, 3), padding = 'same', kernel_initializer = sobelFilterKernel2, input_shape=input_shape[1:])(picture)
    return channel1, channel2

f = lambda x : np.power(x, 2)
g = lambda x : np.sqrt(x)

def joinChannels(channel1, channel2) :
    return g(f(channel1) + f(channel2))

def filterReLU(picture, level) :
    shape = picture.shape
    length = np.prod(shape)
    picture = picture.reshape((length))
    picture = np.where(picture - level > 0, picture, 0)
    maxB = max(1, np.max(picture))
    return picture.reshape(shape) / maxB

def voteKernel(shape, dtype = np.float64) :
    kernel = np.zeros((shape[0], shape[1]))
    for i in range(shape[0]) :
        kernel[i][0] = 1
        kernel[i][shape[1] - 1] = 1
    for j in range(shape[1]) :
        kernel[0][j] = 1
        kernel[shape[0] - 1][j] = 1
    return kernel.reshape(shape)

def upscale(picture) :
    picture = np.array(picture)
    shape = picture.shape
    picture = picture.reshape((shape[1], shape[2], 1))
    picture = tf.keras.layers.Resizing(width, heigth, interpolation = "gaussian")(picture)
    shape = picture.shape
    return picture

def voting(picture, coeff1 = 10.0, coeff2 = 75.0) :
    input_shape = np.shape(picture)
    pictures = []
    toDraw = [16, 21, 27, 32]
    for i in range(16, 80, 4) :
        result = tf.keras.layers.Conv2DTranspose(filters = 1, strides = 1, kernel_size = (i, i), padding = 'same', kernel_initializer = voteKernel, input_shape=input_shape[1:])(picture)
        # if i in toDraw : drawImage(result, shape = (np.shape(result)[1], np.shape(result)[2]))
        result = filterReLU(np.array(result), coeff1 + i * i / coeff2) 
        # if i in toDraw :  drawImage(result, shape = (np.shape(result)[1], np.shape(result)[2]))
        result = tf.keras.layers.Conv2DTranspose(filters = 1, strides = 1, kernel_size = (i, i), padding = 'same', kernel_initializer = voteKernel, input_shape=input_shape[1:])(result)
        # if i in toDraw :  drawImage(result, shape = (np.shape(result)[1], np.shape(result)[2]))
        pictures.append(np.array(upscale(result)).reshape(width, heigth))
    return pictures

def joinPictures(pictures, picture) :
    sumPicture = np.zeros((width, heigth))
    for pic in pictures : 
        sumPicture += pic
    drawImage(sumPicture, shape = (np.shape(sumPicture)[0], np.shape(sumPicture)[1]))
    result = np.zeros((width, heigth, 3))
    for i in range(width) : 
        for j in range(heigth) :
            result[i][j][0] = max(0, 255 * picture[i][j][0] - sumPicture[i][j])
            result[i][j][2] = max(0, min(255, 255 * picture[i][j][2] + 2 * sumPicture[i][j]))
            result[i][j][1] = max(0, 255 * picture[i][j][1] - sumPicture[i][j])
    result.astype(np.int8)
    print(result)
    return result

if __name__ == '__main__' :
    picture = readImage()
    memPicture = np.array(picture)
    memPicture1 = np.array(picture)
    picture = toGrey(picture)
    # drawImage(picture, shape = (width, heigth), rgb = False, scale = 255)
    picture = pool(picture)
    # drawImage(picture, shape = (np.shape(picture)[1], np.shape(picture)[2]))
    picture = gaussianBlur(picture)
    # drawImage(picture, shape = (np.shape(picture)[1], np.shape(picture)[2]))
    channel1, channel2 = sobelFilter(picture)
    # drawImage(channel1, shape = (np.shape(channel1)[1], np.shape(channel1)[2]))
    # drawImage(channel2, shape = (np.shape(channel2)[1], np.shape(channel2)[2]))
    picture = joinChannels(channel1, channel2)
    # drawImage(picture, shape = (np.shape(picture)[1], np.shape(picture)[2]))
    picture = filterReLU(picture, 0.9)
    # drawImage(picture, shape = (np.shape(picture)[1], np.shape(picture)[2]))
    pictures = voting(picture)
    picture = joinPictures(pictures, memPicture)
    image = Image.fromarray(picture.astype(np.int8), mode = 'RGB').show() 
