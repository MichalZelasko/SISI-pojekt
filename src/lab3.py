import tensorflow as tf
import numpy as np
import random

def reformat(labels) :
    array = np.zeros((labels.shape[0], 10))
    i = 0
    for label in labels :
        array[i][label[0]] = 1
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

def optimizeModel(model, x_train, y_train) :
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss=tf.keras.losses.CategoricalCrossentropy(), metrics = ['accuracy'], jit_compile = True)
    model.fit(x = x_train, y = y_train, batch_size = 64, epochs = 1, validation_split = 0.2, use_multiprocessing = True)
    return model

def getWeights(model) :
    sum = 0
    for layer in model.layers :
        sum += len(layer.weights)
    return sum

def testA(x_train, y_train, x_test, y_test, filter_number) :
    model = createModel(filter_number)
    model = optimizeModel(model, x_train, y_train)
    print(f"Total weights number: {getWeights(model)}")
    model.evaluate(x = x_test, y = y_test)

def testB(x_train, y_train, x_test, y_test, blockNumber = 2) :
    model = createBlockModel(blockNumber)
    model = optimizeModel(model, x_train, y_train)
    print(f"Total weights number: {getWeights(model)}")
    model.evaluate(x = x_test, y = y_test)

def testC(x_train, y_train, x_test, y_test, blockNumber = 2) :
    model = createBlockModel(blockNumber, function = modelBatchNormalization)
    model = optimizeModel(model, x_train, y_train)
    print(f"Total weights number: {getWeights(model)}")
    model.evaluate(x = x_test, y = y_test)

def testD(x_train, y_train, x_test, y_test, blockNumber = 4) :
    model = createBlockModel(blockNumber, function = modelBatchNormalization, dropout = True)
    model = optimizeModel(model, x_train, y_train)
    print(f"Total weights number: {getWeights(model)}")
    model.evaluate(x = x_test, y = y_test)

def modifyImage(images) :
    result = []
    for image in images :
        shape = image.shape
        width, heigth = shape[0] + 2 * (random.randint(1, 5) - 3), shape[1] + 2 * (random.randint(1, 5) - 3)
        result.append(tf.keras.layers.Resizing(width, heigth, interpolation = "gaussian")(image))
    return result

def testE(x_train, y_train, x_test, y_test, blockNumber = 4) :
    model = createBlockModel(blockNumber, function = modelBatchNormalization, dropout = True, GAP = True)
    model = optimizeModel(model, x_train, y_train)
    print(f"Total weights number: {getWeights(model)}")
    model.evaluate(x = x_test, y = y_test)
    # model.predict(modifyImage(x_test[0:1]))

def test(function, x_train, y_train, x_test, y_test, argument5, testNumber) :
    print(f"\n\n\nTest {testNumber}==============\n\n\n")
    function(x_train, y_train, x_test, y_test, argument5)
    print(f"\n\n\nEnd==============\n\n\n")

if __name__ == "__main__" :
    x_train, y_train, x_test, y_test, y_train_basic, y_test_basic = loadData()

    test(testA, x_train, y_train, x_test, y_test, 5,  1)
    test(testA, x_train, y_train, x_test, y_test, 20, 2)
    test(testB, x_train, y_train, x_test, y_test, 2,  3)
    test(testB, x_train, y_train, x_test, y_test, 4,  4)
    test(testC, x_train, y_train, x_test, y_test, 4,  5)
    test(testD, x_train, y_train, x_test, y_test, 4,  6)
    # test(testE, x_train, y_train, x_test, y_test, 4,  7)
    


