import matplotlib.pyplot as plt
import tensorflow.keras as K
import tensorflow as tf
import splitfolders
import numpy as np
from sklearn.metrics import classification_report
import sys
import tqdm

np.set_printoptions(threshold=sys.maxsize)

# CLASSES
class_names_master = ["APPLE_BLACK_ROT", "APPLE_CEDAR_RUST", "APPLE_HEALTHY", "APPLE_SCAB", "BACKGROUND",
                      "BELLPEPPER_BACTERIAL_SPOT",
                      "BELLPEPPER_HEALTHY", "CHERRY_HEALTHY", "CHERRY_POWDERY_MILDEW", "CITRUS_BLACK_SPOT",
                      "CITRUS_CANKER", "CITRUS_GREENING", "CITRUS_HEALTHY", "CORN_COMMON_RUST", "CORN_GREY_LEAF_SPOT",
                      "CORN_HEALTHY", "CORN_NORTHERN_LEAF_BLIGHT", "GRAPE_BLACK_MEASLES", "GRAPE_BLACK_ROT",
                      "GRAPE_HEALTHY", "GRAPE_ISARIOPSIS_LEAF_SPOT", "PEACH_BACTERIAL_SPOT", "PEACH_HEALTHY",
                      "POTATO_EARLY_BLIGHT", "POTATO_HEALTHY", "POTATO_LATE_BLIGHT", "STRAWBERRY_HEALTHY",
                      "STRAWBERRY_LEAF_SCORCH", "TOMATO_BACTERIAL_SPOT", "TOMATO_EARLY_BLIGHT", "TOMATO_HEALTHY",
                      "TOMATO_LATE_BLIGHT", "TOMATO_LEAF_MOLD", "TOMATO_MOSAIC_VIRUS", "TOMATO_SEPTORIA_LEAF_SPOT",
                      "TOMATO_SPIDER_MITES", "TOMATO_TARGET_SPOTS", "TOMATO_YELLOW_LEAF_CURL_VIRUS"]
# CONSTANTS
NUM_OUTPUT_CLASSES = 38
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-2
SPLIT_RATIO = (0.9, 0.05, 0.05)
CHOSEN_CLASS = class_names_master
IMG_SHAPE = (224, 224)

# CONSTANTS PATHS
PLANT = "MASTER"
PATH_MODEL = "Models/" + PLANT + "/" + PLANT + "_Model.h5"
PATH_MODEL_FOLDER = "Models/" + PLANT + "/"
DATA_LOCATION = "Datasets/" + PLANT + "_DATASET"
DATA_SPLIT_LOCATION = "Datasets_Split/Split" + PLANT


def get_result(frame, selected_model):
    test_image_array = np.array([frame])
    result = selected_model.predict(test_image_array)
    result = np.argmax(result)
    return result


def split():
    # split data into train, val and test
    splitfolders.ratio(DATA_LOCATION, output=DATA_SPLIT_LOCATION, ratio=SPLIT_RATIO)


def testing(NUM_TEST_IMAGES):
    # load datagen
    testing_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # load test datasets
    test_it = testing_datagen.flow_from_directory(DATA_SPLIT_LOCATION + "/test", class_mode='sparse',
                                                  batch_size=NUM_TEST_IMAGES,
                                                  target_size=IMG_SHAPE)
    # load test model
    test_model = K.models.load_model(PATH_MODEL)

    # generate batch and plot prediction results
    imgs, label = test_it.next()
    for i in range(9):
        pred = get_result(imgs[i], test_model)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[i].astype('uint8'))
        plt.subplots_adjust(hspace=0.5)
        plt.subplots_adjust(wspace=0.5)
        plt.title("Actual: " + CHOSEN_CLASS[int(label[i])] + "\n" + "Prediction: " + CHOSEN_CLASS[pred], fontsize=8)
        plt.axis("off")
    plt.savefig(PATH_MODEL_FOLDER + PLANT + "_PREDICTIONS.png", dpi=300, bbox_inches='tight')
    plt.show()

    # calculate overall accuracy of model on test data set and create prediction arrays
    score = 0
    y_actual = []
    y_pred = []
    for i in range(NUM_TEST_IMAGES):
        pred = get_result(imgs[i], test_model)
        y_actual.append(CHOSEN_CLASS[int(label[i])])
        y_pred.append(CHOSEN_CLASS[pred])
        if pred == int(label[i]):
            score += 1

    acc = score * 100 / len(imgs)
    print(round(acc, 4))

    # Generate the confusion matrix and plot
    print(y_actual)
    print(y_pred)

    print(classification_report(y_actual, y_pred, target_names=class_names_master))


def model_maker():
    # define input new input layer, load resnet50 model
    input_t = K.Input(shape=(224, 224, 3))
    resModel = K.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=input_t)

    # freeze layers in resnet (up to stage5) except batch normal layers (not all layers)
    for i, layer in enumerate(resModel.layers[:143]):
        if "_bn" in layer.name:
            pass
        else:
            layer.trainable = False

    # define optimiser
    opt = K.optimizers.Adam(learning_rate=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)

    # define model
    model = K.models.Sequential()
    model.add(resModel)
    model.add(K.layers.AveragePooling2D(pool_size=(7, 7)))
    model.add(K.layers.Flatten())
    model.add(K.layers.Dense(256, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.Dense(NUM_OUTPUT_CLASSES, activation='softmax'))
    model.compile(optimizer=opt, loss=K.losses.SparseCategoricalCrossentropy(), metrics="sparse_categorical_accuracy")
    model.summary()
    return model


def train_model():
    # load model
    model = model_maker()
    # define train datagen
    training_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=90,
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        width_shift_range=[-10, 10],
        height_shift_range=[-10, 10],
        zoom_range=[0.8, 1.2],
        shear_range=0.1)

    # define val datagen
    validating_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # load train and val datasets
    train_it = training_datagen.flow_from_directory(DATA_SPLIT_LOCATION + "/train", class_mode='sparse',
                                                    batch_size=BATCH_SIZE,
                                                    target_size=IMG_SHAPE)
    val_it = validating_datagen.flow_from_directory(DATA_SPLIT_LOCATION + "/val", class_mode='sparse',
                                                    batch_size=BATCH_SIZE,
                                                    target_size=IMG_SHAPE)
    test_it = validating_datagen.flow_from_directory(DATA_SPLIT_LOCATION + "/test", class_mode='sparse',
                                                     batch_size=BATCH_SIZE,
                                                     target_size=IMG_SHAPE)

    # generate one batch from iterator and check properties
    batchX, batchy = train_it.next()
    print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

    # define train and val steps from data size and batch size
    STEP_SIZE_TRAIN = train_it.n // train_it.batch_size
    STEP_SIZE_VALID = val_it.n // val_it.batch_size

    print(STEP_SIZE_TRAIN)
    print(STEP_SIZE_VALID)

    # generate augmented batch and show images with class name
    img, label = train_it.next()
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(img[i].astype('uint8'))
        plt.title(CHOSEN_CLASS[int(label[i])], fontsize=8)
        plt.subplots_adjust(hspace=0.5)
        plt.axis("off")
    plt.savefig(PATH_MODEL_FOLDER + PLANT + "_TRAIN_IMAGES.png", dpi=300, bbox_inches='tight')
    plt.show()

    # train model
    history = model.fit(
        train_it,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val_it,
        validation_steps=STEP_SIZE_VALID,
        epochs=EPOCHS,
        verbose=1)

    # save model
    model.save(PATH_MODEL)

    # plot accuracy
    plt.plot(history.history['sparse_categorical_accuracy'], marker='x')
    plt.plot(history.history['val_sparse_categorical_accuracy'], marker='x')
    plt.legend(['Training_Accuracy', 'Validation_Accuracy'])
    plt.xticks(range(0, EPOCHS))
    plt.title('Training and Validation Accuracy Graph')
    plt.xlabel('Epochs')
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(PATH_MODEL_FOLDER + PLANT + "_ACC.png", dpi=300, bbox_inches='tight')
    plt.show()

    # plot loss
    plt.plot(history.history['loss'], marker='x')
    plt.plot(history.history['val_loss'], marker='x')
    plt.legend(['Training_Loss', 'Validation Loss'])
    plt.xticks(range(0, EPOCHS))
    plt.title('Training and Validation Loss Graph')
    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(PATH_MODEL_FOLDER + PLANT + "_LOSS.png", dpi=300, bbox_inches='tight')
    plt.show()
    return test_it.n


def main():
    split()
    test_imgs = train_model()
    testing(test_imgs)


if __name__ == '__main__':
    main()
