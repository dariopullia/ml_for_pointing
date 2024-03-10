import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import hyperopt as hp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import healpy as healpy

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
    # remove images where np.sum is 655350000
    print("Removing corrupted images...")
    corrupted_images = []
    for i in range(dataset_img.shape[0]):
        if (np.sum(dataset_img[i])==655350000):
            corrupted_images.append(i)
    print("Corrupted images: ", len(corrupted_images))
    dataset_img = np.delete(dataset_img, corrupted_images, axis=0)
    dataset_label = np.delete(dataset_label, corrupted_images, axis=0)
    

    print("Dataset loaded.")
    print("Dataset_img shape: ", dataset_img.shape)
    print("Dataset_lab shape: ", dataset_label.shape)
    print("dataset_img type: ", type(dataset_img))
    print("dataset_lab type: ", type(dataset_label))

    if not os.path.exists(output_folder+"samples/"):
        os.makedirs(output_folder+"samples/")
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
    save_samples_from_ds(dataset_img, dataset_label, output_folder+"samples/")
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
    print("Dataset splitted.")

    if aug_coefficient>1:
        print("Data augmentation...")
        print("Train images shape before: ", train_images.shape)
        train_images, train_labels = data_augmentation(train_images, train_labels, coefficient=aug_coefficient, prob_per_flip=prob_per_flip)
        print("Train images shape after: ", train_images.shape)
        print("Data augmented.")
    
    print("Unique labels: ", np.unique(train_labels, return_counts=True))

    # Create the datasets
    print("Creating the dataset objects...")
    with tf.device("CPU"):
        train = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(10000).batch(32)
        validation = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels)).batch(32)
        test = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)
    print("Datasets created.")
    return train, validation, test

def data_augmentation(dataset, labels, coefficient=2, prob_per_flip=0.5):
    # create the augmented dataset
    augmented_dataset = []
    augmented_labels = []
    # for each sample
    for i in range(int(coefficient*len(dataset))):
        index = np.random.randint(0, len(dataset))
        # get the sample
        sample = dataset[index]
        # get the label
        label = labels[index]
        if np.random.rand() > prob_per_flip:
            # flip the sample
            sample = np.flipud(sample)
        if np.random.rand() > prob_per_flip:
            sample = np.fliplr(sample)
        augmented_dataset.append(sample)
        augmented_labels.append(label)
    return np.array(augmented_dataset), np.array(augmented_labels)

def calculate_metrics(y_true, y_pred,):
    # calculate the confusion matrix, the accuracy, and the precision and recall 
    # binary trick
    y_pred_am = np.where(y_pred > 0.5, 1, 0)
    cm = confusion_matrix(y_true, y_pred_am, normalize='true')
    # compute precision matrix
    

    accuracy = accuracy_score(y_true, y_pred_am)
    precision = precision_score(y_true, y_pred_am, average='macro')
    recall = recall_score(y_true, y_pred_am, average='macro')
    f1 = f1_score(y_true, y_pred_am, average='macro')

    return cm, accuracy, precision, recall, f1
    
def log_metrics(y_true, y_pred, output_folder="", label_names=["CC", "ES"]):
    cm, accuracy, precision, recall, f1 = calculate_metrics(y_true, y_pred)
    print("Confusion Matrix")
    print(cm)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_folder+f"metrics.txt", "a") as f:
        f.write("Confusion Matrix\n")
        f.write(str(cm)+"\n")
        f.write("Accuracy: "+str(accuracy)+"\n")
        f.write("Precision: "+str(precision)+"\n")
        f.write("Recall: "+str(recall)+"\n")
        f.write("F1: "+str(f1)+"\n")
    # save confusion matrix 
    plt.figure(figsize=(10,10))
    plt.title("Confusion matrix", fontsize=28)
    sns.heatmap(cm, annot=True, cmap="YlGnBu", xticklabels=label_names, yticklabels=label_names, annot_kws={"fontsize": 20})
    plt.ylabel('True label', fontsize=28)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Predicted label', fontsize=28)
    plt.savefig(output_folder+f"confusion_matrix.png")
    plt.clf()
    # Binarize the output
    y_test = label_binarize(y_true, classes=np.arange(len(label_names)))
    n_classes = y_test.shape[1]
    
    plt.figure()

    fpr, tpr, _ = roc_curve(y_true[:], y_pred[:])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = {0:0.2f})'.format(roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=28)
    plt.ylabel('True Positive Rate', fontsize=28)
    plt.title('ROC curve', fontsize=28)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(output_folder+"roc_curve.png")
    plt.clf()
    # create an histogram of the predictions
    print(y_pred.shape)
    print(y_true.shape)
    y_true = np.reshape(y_true, (y_true.shape[0],))
    bkg_preds = y_pred[y_true < 0.5]
    sig_preds = y_pred[y_true > 0.5]
    print("Background predictions: ", bkg_preds.shape)
    print("Signal predictions: ", sig_preds.shape)

    plt.hist(bkg_preds, bins=50, alpha=0.5, label=f'{label_names[0]} (n={bkg_preds.shape[0]})')
    plt.hist(sig_preds, bins=50, alpha=0.5, label=f'{label_names[1]} (n={sig_preds.shape[0]})')
    plt.legend(loc='upper right')
    plt.xlabel('Prediction')
    plt.ylabel('Counts')
    plt.title('Predictions')
    plt.savefig(output_folder+f"predictions.png")
    plt.clf()

def save_sample_img(ds_item, output_folder, img_name):
    if ds_item.shape[2] == 1:
        img = ds_item[:,:,0]
        plt.figure(figsize=(10, 26))
        plt.title(img_name)
        plt.imshow(img)
        # plt.colorbar()
        # add x and y labels
        plt.xlabel("Channel")
        plt.ylabel("Time (ticks)")
        # save the image, with a bbox in inches smaller than the default but bigger than tight
        plt.savefig(output_folder+img_name+".png", bbox_inches='tight', pad_inches=1)
        plt.close()

    else:
        img_u = ds_item[:,:,0]
        img_v = ds_item[:,:,1]
        img_x = ds_item[:,:,2]
        fig = plt.figure(figsize=(8, 20))
        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                        nrows_ncols=(1,3),
                        axes_pad=0.5,
                        share_all=True,
                        cbar_location="right",
                        cbar_mode="single",
                        cbar_size="30%",
                        cbar_pad=0.25,
                        )   


        if img_u[0, 0] != -1:
            im = grid[0].imshow(img_u)
            grid[0].set_title('U plane')
        if img_v[0, 0] != -1:
            im = grid[1].imshow(img_v)
            grid[1].set_title('V plane')
        if img_x[0, 0] != -1:
            im = grid[2].imshow(img_x)
            grid[2].set_title('X plane')
        grid.cbar_axes[0].colorbar(im)
        grid.axes_llc.set_yticks(np.arange(0, img_u.shape[0], 100))
        # save the image
        plt.savefig(output_folder+ 'multiview_' + img_name + '.png')
        plt.close()

def save_samples_from_ds(dataset, labels, output_folder, name="img", n_samples_per_label=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # get the labels
    labels_unique = np.unique(labels, return_counts=True)
    # get the samples
    for label in labels_unique[0]:
        # get the indices
        indices = np.where(labels == label)[0]
        indices = indices[:np.minimum(n_samples_per_label, indices.shape[0])]
        samples = dataset[indices]
        # save the images
        for i, sample in enumerate(samples):
            save_sample_img(sample, output_folder, name+"_"+str(label)+"_"+str(i))
    # make one image for each label containing n_samples_per_label images, using plt suplot
    fig= plt.figure(figsize=(20, 20))
    for i, label in enumerate(labels_unique[0]):
        grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                nrows_ncols=(1,10),
                axes_pad=0.5,
                share_all=True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="30%",
                cbar_pad=0.25,
                )   
        indices = np.where(labels == label)[0]
        indices = indices[:np.minimum(n_samples_per_label, indices.shape[0])]
        samples = dataset[indices]
        # save the images
        plt.suptitle("Label: "+str(label), fontsize=25)

        for j, sample in enumerate(samples):
            im = grid[j].imshow(sample[:,:,0])
            grid[j].set_title(j)

        grid.cbar_axes[0].colorbar(im)
        grid.axes_llc.set_yticks(np.arange(0, sample.shape[0], 100))
        plt.savefig(output_folder+ 'all_'+str(label)+'.png')
        plt.clf()

def test_model(model, test, output_folder, label_names=["CC", "ES"]):
    print("Doing some test...")
    predictions = model.predict(test)      
    # Calculate metrics
    print("Calculating metrics...")
    # get the test labels from the test dataset
    test_labels = np.array([label for _, label in test], dtype=object)
    test_labels = np.concatenate(test_labels, axis=0)
    log_metrics(test_labels, predictions, output_folder=output_folder, label_names=label_names)
    print("Metrics calculated.")
    print("Drawing model...")
    keras.utils.plot_model(model, output_folder+"architecture.png", show_shapes=True)
    print("Model drawn.")
    print("Drawing histogram of energies...")
    test_img = np.array([img for img, _ in test], dtype=object)
    test_img = np.concatenate(test_img, axis=0)
    
    histogram_of_enegies(test_labels, predictions, test_img, limit=0.5, output_folder=output_folder)

    print("Test done.")

def histogram_of_enegies(test_labels, predictions, images, limit=0.5, output_folder=""):
    # check if some images are corrupted
    corrupted_images = []
    for i in range(images.shape[0]):
        if (np.sum(images[i])==0):
            corrupted_images.append(i)
    print("Corrupted images: ", len(corrupted_images))


    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    all_images = []
    for i in range(len(test_labels)):
        if test_labels[i] == 1 and predictions[i] > limit:
            true_positives.append(np.sum(images[i]))
        elif test_labels[i] == 0 and predictions[i] < limit:
            true_negatives.append(np.sum(images[i]))
        elif test_labels[i] == 0 and predictions[i] > limit:
            false_positives.append(np.sum(images[i]))
        elif test_labels[i] == 1 and predictions[i] < limit:
            false_negatives.append(np.sum(images[i]))
        all_images.append(np.sum(images[i]))
    
    print("True Positives: ", len(true_positives))
    print("True Negatives: ", len(true_negatives))
    print("False Positives: ", len(false_positives))
    print("False Negatives: ", len(false_negatives))
    print("All images: ", len(all_images))
    print(np.unique(np.array(all_images), return_counts=True))

    # sum the pixel values
    plt.figure()
    plt.hist(true_positives, range=(0, 3e6), bins=50, alpha=0.5, label='True Positives (n='+str(len(true_positives))+')')
    plt.hist(true_negatives, range=(0, 3e6), bins=50, alpha=0.5, label='True Negatives (n='+str(len(true_negatives))+')')
    plt.hist(false_positives, range=(0, 3e6), bins=50, alpha=0.5, label='False Positives (n='+str(len(false_positives))+')')
    plt.hist(false_negatives, range=(0, 3e6), bins=50, alpha=0.5, label='False Negatives (n='+str(len(false_negatives))+')')

    plt.legend(loc='upper right')
    plt.xlabel('Pixel value')
    plt.ylabel('Counts')
    plt.title('Pixel value histogram')
    plt.savefig(output_folder+"pixel_value_histogram.png")
    plt.clf()
