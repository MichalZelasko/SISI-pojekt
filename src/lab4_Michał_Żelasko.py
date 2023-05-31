import tensorflow as tf
import numpy as np
from PIL import Image
from jmd_imagescraper.core import *
import matplotlib.pyplot as plt
from pathlib import Path
import os

def importPictures(categories, number = 300) :
    root = Path().cwd()/"images"
    for category in categories :
        duckduckgo_search(root, category, category, max_results = number)
    print("Downloading completed")

def chooseAndFormat(categories, basic_path = "/content/images/", short_path = "/content/", number = 300) :
    os.mkdir("/content/dataset")
    for category in categories :
        path = basic_path + category
        elements, count = os.listdir(path), 0
        os.mkdir("/content/dataset/" + category)
        for element in elements :
            if element[-4:] == ".jpg" and count < number:
                image_path = basic_path + category + "/" + element
                image = tf.keras.preprocessing.image.load_img(image_path)
                imageArray = tf.keras.preprocessing.image.img_to_array(image)
                effect = np.array(tf.image.resize(imageArray, [32, 32], method = "lanczos3"))
                im = Image.fromarray(np.uint8(effect), 'RGB')
                count += 1
                im.save(short_path + "dataset/" + category + "/" + "picture" + str(count) + ".jpg")
                
def createDataset(filepath, height, width) :
    return tf.keras.utils.image_dataset_from_directory(filepath, validation_split = 0.2, subset = "training", seed = 123, image_size = (height, width), batch_size = 480), tf.keras.utils.image_dataset_from_directory(filepath, validation_split = 0.2, subset = "validation", seed = 123, image_size = (height, width), batch_size = 120)

def reformat(labels, number = 10) :
    array = np.zeros((labels.shape[0], number))
    i = 0
    for label in labels :
        array[i][label] = 1
        i += 1
    return array

def loadData() :
    (x_train, y_train_basic), (x_test, y_test_basic) = tf.keras.datasets.cifar10.load_data()
    y_train = reformat(y_train_basic)
    y_test = reformat(y_test_basic)
    return x_train, y_train, x_test, y_test, y_train_basic, y_test_basic

def printShape(x_train, y_train, x_test, y_test) :
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

def createModel(filter_number) :
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape = (32, 32, 3)),
            tf.keras.layers.Conv2D(filters = filter_number, strides = 1, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid'),
            tf.keras.layers.Conv2D(filters = filter_number, strides = 1, kernel_size = (3, 3), padding = 'same', activation = 'sigmoid'),
            tf.keras.layers.MaxPool2D(pool_size = (8, 8), strides = 8),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units = 10, activation = 'softmax')
        ]
    )
    return model

def modelBlock(filter_number, activation = 'sigmoid') :
    return [
        tf.keras.layers.Conv2D(filters = filter_number, strides = 1, kernel_size = (3, 3), padding = 'same', activation = activation),
        tf.keras.layers.Conv2D(filters = filter_number, strides = 1, kernel_size = (3, 3), padding = 'same', activation = activation),
        tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = 2)
    ]

def modelBatchNormalization(filter_number, activation = 'sigmoid') :
    return [
        tf.keras.layers.Conv2D(filters = filter_number, strides = 1, kernel_size = (3, 3), padding = 'same', activation = activation),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters = filter_number, strides = 1, kernel_size = (3, 3), padding = 'same', activation = activation),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = 2)
    ]

def createBlockModel(blockNumber = 2, function = modelBlock, dropout = False, GAP = False) :
    if GAP : layers = [tf.keras.layers.Input((None, None, 3))]
    else : layers = [tf.keras.layers.Input(shape = (32, 32, 3))]
    startSize = 20
    dropoutStart = 0.1
    for _ in range(blockNumber) :
        layers.extend(function(startSize, activation='relu'))
        if dropout : layers.append(tf.keras.layers.Dropout(rate = dropoutStart))
        startSize *= 2
        dropoutStart += 0.1
    if GAP : layers[-1] = tf.keras.layers.GlobalAveragePooling2D()
    layers.extend([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units = 10, activation = 'softmax')
    ])
    return tf.keras.Sequential(layers)

def optimizeModel(model, x_train, y_train, epochs = 150) :
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss=tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'], jit_compile = True)
    history = model.fit(x = x_train, y = y_train, batch_size = 64, epochs = epochs, validation_split = 0.2, use_multiprocessing = True)
    return model, history

def plotHistory(history) :
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower left')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0,1.1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(np.array(val_loss) / 10, label='Validation Loss')
    plt.legend(loc='lower left')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()


def test(model, x_train, y_train, x_test, y_test, doPlot = False, doEvaluate = True, epochs = 150) :
    model, history = optimizeModel(model, x_train, y_train, epochs = epochs)
    if doEvaluate : model.evaluate(x = x_test, y = y_test)
    if doPlot : plotHistory(history)
    return model

def loadDataset(directory, size = 32, number = 10) :
    training, validation = createDataset(directory, size, size)
    for x_train, y_train_basic in training :
        break
    for x_test, y_test_basic in validation :
        break
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = reformat(np.array(y_train_basic), number)
    y_test = reformat(np.array(y_test_basic), number)
    return x_train, y_train, x_test, y_test, y_train_basic, y_test_basic

def pretrain() :
    x_train, y_train, x_test, y_test, _, _ = loadData()
    model = createBlockModel(4, function = modelBatchNormalization, dropout = True, GAP = True)
    model = test(model, x_train, y_train, x_test, y_test, doEvaluate = False)
    return model

def prepare(directory, model, size = 32) :
    model.layers[-1] = tf.keras.layers.Dense(units = 10, activation = 'softmax')
    for layer in model.layers[:-1] :
        layer.trainable = False
    x_train, y_train, x_test, y_test, _, _ = loadDataset(directory, size)
    return test(model, x_train, y_train, x_test, y_test, doPlot = True, doEvaluate = True)

def prepareXception(directory, model, doFreeze = True, epochs = 150) :
    for layer in model.layers :
      if doFreeze or layer.name[:5] == "batch" :
          layer.trainable = False
      else :
          layer.trainable = True
    if doFreeze :
        model.layers[-2] = tf.keras.layers.GlobalAveragePooling2D()
        model.layers[-1] = tf.keras.layers.Dense(units = 10, activation = 'softmax')
    model.layers[-2].trainable = True
    model.layers[-1].trainable = True
    
    x_train, y_train, x_test, y_test, _, y_test_basic = loadDataset(directory, 299, 1000)
    return test(model, x_train, y_train, x_test, y_test, doPlot = True, doEvaluate = True, epochs = epochs), x_test, y_test

def writeImage(output) :
    for i in range(10) :
        channel = np.zeros((10, 10))
        for j in range(10) :
            for k in range(10) :
                channel[j][k] = output[0][j][k][i]
        plt.imshow(channel)
        plt.show()

def writeImageWeights(output, weights) :
    result = np.zeros((10, 10))
    for i in range(2048) :
        if i % 21 == 1 : print(f"{100 * i / 2048}%")
        # print(output[0][:][:][i] * np.max(weights[i]))
        # result += output[i] * np.max(weights[i])
        m = np.max(weights[i])
        for j in range(10) :
            for k in range(10) :
                result[j][k] += output[j][k][i] * m
    return result

def rearrange(matrix) :
    result = np.zeros((2048, 10, 10))
    for i in range(2048) :
        print(f"{100 * i / 2048}%")
        for j in range(10) :
            for k in range(10) :
                result[i][j][k] = matrix[0][j][k][i]
    r2 = matrix[0].numpy().reshape((2048, 10, 10))
    print(result - r2)
    return matrix

def checkPrediction(model, prediction, expected, x_test) :
    i, result_ok, result_wrong = 0, [], []
    for pred in prediction :
        if np.argmax(pred) == np.argmax(expected[i]) : result_ok.append(i)
        else : result_wrong.append(i)
        i += 1
    
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    w = model.layers[-1].weights
    
    x = x_test[result_ok[0]:result_ok[0]+1]
    output = model2(x)
    writeImage(output)
    print("OK")
    for i in range(5, min(len(result_ok), 5)) : 
        plt.imshow(x_test[result_ok[i]] / 255)
        plt.show()
        x = x_test[result_ok[i]:result_ok[i]+1]
        out = model2(x)
        plt.imshow(writeImageWeights(out[0], w[0]))
        plt.show()
    print("Wrong")
    for i in range(4, min(len(result_wrong), 6)) : 
        plt.imshow(x_test[result_wrong[i]] / 255)
        plt.show()
        x = x_test[result_wrong[i]:result_wrong[i]+1]
        out = model2(x)
        plt.imshow(writeImageWeights(out[0], w[0]))
        plt.show()
    plt.show()
    
    
    # writeImage(out)
    # print(out.get_shape())
    return result_ok, result_wrong

def mainDataset() :
    categories = ["gingerbread", "apple pie", "european fir"]
    importPictures(categories=categories)
    chooseAndFormat(categories=categories, number = 200)

def step1(directory) :
    x_train, y_train, x_test, y_test, _, _ = loadDataset(directory)
    model = createBlockModel(4, function = modelBatchNormalization, dropout = True, GAP = True)
    model = test(model, x_train, y_train, x_test, y_test, doPlot = True)
    return model, y_test, x_test

def step2(directory) :
    model = pretrain()
    prepare(directory, model)

def step3(directory, doFreeze = True, epochs = 40, doPredict = True) :
    model = tf.keras.applications.Xception(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation="softmax")
    model, x_test, y_test = prepareXception(directory, model, doFreeze, epochs)
    if doPredict : print(checkPrediction(model.predict(x_test), y_test, x_test))
    return model, y_test, x_test

model, y_test, x_test = None, None, None

if __name__ == "__main__" :
    # mainDataset()

    # directory = "dataset256"
    # model, y_test, x_test = step1(directory)
    # step2(directory)
    directory2 = "dataset256"
    model, y_test, x_test = step3(directory2, doFreeze = True, epochs = 75)
    # model, x_test, y_test = prepareXception(directory2, model, doFreeze = True, epochs = 75)
    # print(checkPrediction(model.predict(x_test), y_test, x_test))