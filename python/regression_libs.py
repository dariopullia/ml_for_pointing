import sys
import os
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pylab as pylab
import seaborn as sns
#import hyperopt as hp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
#import healpy as healpy

import general_purpose_libs as gpl


def prepare_data(input_data, input_label, dataset_parameters, output_folder):
    # Load the data
    remove_y_direction = dataset_parameters["remove_y_direction"]
    train_fraction = dataset_parameters["train_fraction"]
    val_fraction = dataset_parameters["val_fraction"]
    test_fraction = dataset_parameters["test_fraction"]
    aug_coefficient = dataset_parameters["aug_coefficient"]
    prob_per_flip = dataset_parameters["prob_per_flip"]

    if not os.path.exists(input_data) or not os.path.exists(input_label):
        print(input_data)
        print(input_label)
        print("Exists input data: ", os.path.exists(input_data))
        print("Exists input label: ", os.path.exists(input_label))
        print("Input file not found.")
        exit()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the dataset
    print("Loading the dataset...")
    dataset_img = np.load(input_data, allow_pickle=True)
    dataset_label = np.load(input_label)
    print("Dataset loaded.")
    print("Dataset_img shape: ", dataset_img.shape)
    print("Dataset_lab shape: ", dataset_label.shape)
    print("dataset_img type: ", type(dataset_img))
    print("dataset_lab type: ", type(dataset_label))

    print("SET Y TO 0")
    dataset_label[:,1] = 0
    # Remove the direction
    if remove_y_direction:
        print("Removing the direction...")
        dataset_label = np.delete(dataset_label, 1, axis=1)
    
    # Normalize the labels
    print("Normalizing the labels...")
    if dataset_label.shape[1] == 3:
        dataset_label = dataset_label/np.linalg.norm(dataset_label, axis=1)[:, np.newaxis]
    elif dataset_label.shape[1] == 2:
        r = np.sqrt(dataset_label[:,0]**2 + dataset_label[:,1]**2)
        dataset_label[:,0] = dataset_label[:,0]/r
        dataset_label[:,1] = dataset_label[:,1]/r
    print("Labels normalized.")


    if not os.path.exists(output_folder+"samples/"):
        os.makedirs(output_folder+"samples/")
    
    for i in range(dataset_label.shape[1]):
        plt.hist(dataset_label[:,i], bins=100, label='dir '+str(i))
    plt.legend()
    plt.savefig(output_folder+"samples/label_hist.png")

    # Check if the dimension of images and labels are the same
    if dataset_img.shape[0] != dataset_label.shape[0]:
        print("Error: the dimension of images and labels are not the same.")
        exit()
    # shuffle the dataset
    print("Shuffling the dataset...")
    index = np.arange(dataset_img.shape[0])
    np.random.shuffle(index)
    dataset_img = dataset_img[index]
    dataset_label = dataset_label[index]
    print("Dataset shuffled.")

    # Save some images
    print("Saving some images...")
    save_samples_from_ds(dataset_img, dataset_label, output_folder+"samples/", name="img", n_samples=10)
    save_labels_in_a_map(dataset_label, output_folder+"samples/", name="full_ds")
    print("Images saved.")
    
    # Split the dataset in training, validation and test
    print("Splitting the dataset...")
    training_fraction = train_fraction
    validation_fraction = val_fraction
    test_fraction = test_fraction

    train_images = dataset_img[:int(dataset_img.shape[0]*training_fraction)]
    validation_images = dataset_img[int(dataset_img.shape[0]*training_fraction):int(dataset_img.shape[0]*training_fraction)+int(dataset_img.shape[0]*validation_fraction)]
    test_images = dataset_img[int(dataset_img.shape[0]*training_fraction)+int(dataset_img.shape[0]*validation_fraction):]
    train_labels = dataset_label[:int(dataset_label.shape[0]*training_fraction)]
    validation_labels = dataset_label[int(dataset_label.shape[0]*training_fraction):int(dataset_label.shape[0]*training_fraction)+int(dataset_label.shape[0]*validation_fraction)]
    test_labels = dataset_label[int(dataset_label.shape[0]*training_fraction)+int(dataset_label.shape[0]*validation_fraction):]
    # save test set for further analysis
    np.save(output_folder+"test_images.npy", test_images)
    np.save(output_folder+"test_labels.npy", test_labels)
    print("Dataset splitted.")

    if aug_coefficient>1:
        print("Data augmentation...")
        print("Train images shape before: ", train_images.shape)
        train_images, train_labels = data_augmentation(train_images, train_labels, coefficient=aug_coefficient, prob_per_flip=prob_per_flip)
        print("Train images shape after: ", train_images.shape)
        print("Data augmented.")

    train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
    validation = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels)).batch(32)
    test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

    return train, validation, test

def test_model(model, test, output_folder):
    # Test the model
    print("Doing some test...")
    predictions = model.predict(test)      
    # Calculate metrics
    print("Calculating metrics...")
    # get the test labels from the test dataset
    test_labels = np.array([label for _, label in test], dtype=object)
    test_labels = np.concatenate(test_labels, axis=0)
    log_metrics(test_labels, predictions, output_folder=output_folder)
    print("Metrics calculated.")
    print("Drawing model...")
    keras.utils.plot_model(model, output_folder+"architecture.png", show_shapes=True)
    print("Model drawn.")
    print("Test done.")

def log_metrics(test_labels, predictions, output_folder):
    save_labels_in_a_map(test_labels, output_folder, name="map_true")
    save_labels_in_a_map(predictions, output_folder, name="map_predictions")
    plot_diff(test_labels, predictions, output_folder)

def plot_diff(test_labels, predictions, output_folder):
    # plot the difference between the true and predicted labels
    if test_labels.shape[1] == 3:
        angles_true = from_coordinate_to_theta_phi(test_labels)
        angles_pred = from_coordinate_to_theta_phi(predictions)
        thetas_true, phis_true = angles_true[:,0], angles_true[:,1]
        thetas_pred, phis_pred = angles_pred[:,0], angles_pred[:,1]
        plt.figure(figsize=(10, 10))
        plt.title("Difference in Theta")
        plt.hist(thetas_true-thetas_pred, bins=100, alpha=0.5, label="Theta")
        plt.legend(loc='upper right')
        plt.savefig(output_folder+"theta_diff_hist.png")
        plt.clf()
        plt.figure(figsize=(10, 10))
        plt.title("Difference in Phi")
        plt.hist(phis_true-phis_pred, bins=100, alpha=0.5, label="Phi")
        plt.legend(loc='upper right')
        plt.savefig(output_folder+"phi_diff_hist.png")
        plt.clf()
    elif test_labels.shape[1] == 2:
        x_true, y_true = test_labels[:,0], test_labels[:,1]
        x_pred, y_pred = predictions[:,0], predictions[:,1]
        # normalize the labels
        r = np.sqrt(x_true**2 + y_true**2)
        x_true = x_true/r
        y_true = y_true/r
        r = np.sqrt(x_pred**2 + y_pred**2)
        x_pred = x_pred/r
        y_pred = y_pred/r

        theta_true = np.arctan2(y_true, x_true)
        theta_pred = np.arctan2(y_pred, x_pred)
        plt.figure(figsize=(10, 10))
        plt.title("Difference in Theta")
        plt.hist(theta_true-theta_pred, bins=100, alpha=0.5, label="Theta")
        plt.legend(loc='upper right')
        plt.savefig(output_folder+"theta_diff_hist.png")
        plt.clf()      

def save_labels_in_a_map(dataset_label, output_folder, name="map"):
    if dataset_label.shape[1] == 3:

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # create an image with healpy, each label is a direction in x,y,z
        nside = 32
        npix = healpy.nside2npix(nside)
        # create a map with the number of pixels
        map = np.zeros(npix)
        # create a map with the number of pixels

        angles = from_coordinate_to_theta_phi(dataset_label)
        thetas, phis = angles[:,0], angles[:,1]

        plt.figure(figsize=(10, 10))
        plt.title("Labels map")
        thetas = np.mod(thetas, np.pi)
        phis = np.mod(phis, 2*np.pi)
        # get the indices
        indices = healpy.ang2pix(nside, thetas, phis)
        # fill the map
        for index in indices:
            map[index] += 1
        # plot the map
        plt.figure(figsize=(10, 10))
        plt.title("Labels map")
        plt.hist(thetas, bins=100, alpha=0.5, label="Theta")
        plt.hist(phis, bins=100, alpha=0.5, label="Phi")
        plt.legend(loc='upper right')
        plt.title("Theta and Phi")
        plt.savefig(output_folder+name+"_theta_phi_hist.png")
        plt.clf()

        healpy.mollview(map, title=name, cmap="viridis")
        plt.savefig(output_folder+name+".png")
        plt.close()

    elif dataset_label.shape[1] == 2:
        plt.figure(figsize=(10, 10))
        plt.title("Labels map")
        x_label, y_label = dataset_label[:,0], dataset_label[:,1]
        # normalize the labels
        r = np.sqrt(x_label**2 + y_label**2)
        x_label = x_label/r
        y_label = y_label/r
        theta_label = np.arctan2(y_label, x_label)
        plt.hist(theta_label, bins=50, alpha=0.5, label="Theta")
        plt.legend(loc='upper right')
        plt.savefig(output_folder+name+"_theta_phi_hist.png")
        plt.clf()
        plt.hist(x_label, bins=50, alpha=0.5, label="X")
        plt.hist(y_label, bins=50, alpha=0.5, label="Y")
        plt.legend(loc='upper right')
        plt.savefig(output_folder+name+"_x_y_hist.png")
        plt.clf()    


def save_samples_from_ds(dataset, labels, output_folder, name="img", n_samples=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # get the labels
    fig= plt.figure(figsize=(20, 20))
    for i in range(n_samples):
        # get the sample
        sample = dataset[i]
        # get the label
        label = labels[i]
        # save the sample
        gpl.save_sample_img(sample, output_folder, name+"_"+str(i))
        plt.clf()
    plt.close()
            
def from_coordinate_to_theta_phi(coords):
    # nomalize the coordinates
    coords = coords/np.linalg.norm(coords, axis=1)[:, np.newaxis]
    x, y, z = coords[:,0], coords[:,2], coords[:,1]
    r = np.sqrt(x**2 + y**2 + z**2)
    # theta is in [0, pi]
    theta = np.arccos(z/r)
    # phi is in [-pi, pi]
    phi = np.arctan2(y, x)

    return np.array([theta, phi]).T

def data_augmentation(dataset, labels, coefficient=2, prob_per_flip=0.5):
    # create the augmented dataset
    augmented_dataset = []
    augmented_labels = []
    # for each sample
    if (labels.shape[1]) == 3:
        for i in range(int(coefficient*len(dataset))):
            index = np.random.randint(0, len(dataset))
            # get the sample
            sample = dataset[index]
            # get the label
            label = labels[index]
            new_label = [label[0], label[1], label[2]]
            if np.random.rand() > prob_per_flip:
                # flip the sample
                sample = np.flipud(sample)
                new_label[0] = -label[0]
            if np.random.rand() > prob_per_flip:
                sample = np.fliplr(sample)
                new_label[2] = -label[2]
            augmented_dataset.append(sample)
            augmented_labels.append(new_label)
    elif (labels.shape[1]) == 2:
        for i in range(int(coefficient*len(dataset))):
            index = np.random.randint(0, len(dataset))
            # get the sample
            sample = dataset[index]
            # get the label
            label = labels[index]
            new_label = [label[0], label[1]]
            if np.random.rand() > prob_per_flip:
                # flip the sample
                sample = np.flipud(sample)
                new_label[0] = -label[0]
            if np.random.rand() > prob_per_flip:
                sample = np.fliplr(sample)
                new_label[1] = -label[1]
            augmented_dataset.append(sample)
            augmented_labels.append(new_label)

    return np.array(augmented_dataset), np.array(augmented_labels)

def my_loss_function(y_true, y_pred):
    # the loss is the dot product of the true and predicted values, divided by the product of the norms of the two vectors
    return 1 - tf.reduce_sum(y_true * y_pred, axis=-1) / (tf.norm(y_true, axis=-1) * tf.norm(y_pred, axis=-1))

def my_loss_function_both_dir(y_true, y_pred):
    # the loss accepts two directions, the true and the predicted one
    return tf.reduce_sum(tf.minimum(1 - tf.reduce_sum(y_true * y_pred, axis=-1) / (tf.norm(y_true, axis=-1) * tf.norm(y_pred, axis=-1)), 1 + tf.reduce_sum(y_true * y_pred, axis=-1) / (tf.norm(y_true, axis=-1) * tf.norm(y_pred, axis=-1))), axis=-1)

