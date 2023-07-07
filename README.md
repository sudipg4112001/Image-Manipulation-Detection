# Image-Manipulation-Detection

To develop an image manipulation detection system, specifically focusing on two common types of forgeries: 

`copy-move` and `splicing`.

- **Copy-Move Forgery:** Copy-move forgery refers to the act of copying a specific portion of an image and pasting it onto another area within the same image.
- **Splicing Forgery:** Splicing forgery involves merging two or more images to create a single composite image. This technique aims to combine different elements or regions from multiple sources.

## Dataset Description

The initial dataset contains two directories named `test` and `traindev` for the test and training data, respectively. Each directory contains three classes: `authentic`, `copy-moved`, and `spliced`. The `authentic` class includes original RGB images, while the `copy-moved` and `spliced` classes include RGB images in the `images` folder and masked images in the `masked` folder.
In both the test and training data, each class consists of 166 images in the test data and 1494 images in the training data.

Dataset Structure:

- Dataset                                                                            
  - test
    - authentic
      - [166 RGB images]
    - copy-moved
      - images
        - [166 RGB images]
      - masked
        - [166 masked images]
    - spliced
      - images
        - [166 RGB images]
      - masked
        - [166 masked images]
  - traindev
    - authentic
      - [1494 RGB images]
    - copy-moved
      - images
        - [1494 RGB images]
      - masked
        - [1494 masked images]
    - spliced
      - images
        - [1494 RGB images]
      - masked
        - [1494 masked images]

Whereas, the aim of this task is to categorize 'copy-moved' and 'spliced' images. Initially, the experiment focused on using RGB images for binary classification of these classes. However, the performance of the state-of-the-art approach did not meet the desired level of accuracy. As a result, the approach was modified to classify the masked images provided in the dataset instead, restructing the dataset according the directory structure shown below and the same has been provided in `spoof-binary.zip` folder.

- Dataset
  - test
    - copy-moved
        - [166 masked images]
    - spliced
        - [166 masked images]
  - traindev
    - copy-moved
        - [1494 masked images]
    - spliced
        - [1494 masked images]

## Visualization of the classes

<div style="display:flex;">
  <img src="https://github.com/sudipg4112001/Image-Manipulation-Detection/assets/60208804/f4c5f0e7-871e-4085-b32f-85e59749f4ef" alt="Image 1" width="500">
  <img src="https://github.com/sudipg4112001/Image-Manipulation-Detection/assets/60208804/130a9817-350d-43f0-a531-d7aca80a33bc" alt="Image 2" width="500">
</div>

## Methodology

The dataset includes both training and test data. However, a portion of the training data, specifically 20%, is separated and utilized as a validation set. Hence, the split comes as 2390, 598 and 332 for train, validation and test data respectively. 

The model architecture incorporates InceptionResNetV2 as the backbone with imagenet weights with input dimension of 224x224x3, where 224x224 signifies the image dimension and 3 signifies the image texture(3 represents RGB). This is succeeded by a BatchNormalization layer, a Dense layer(D1), a Dropout layer, and another Dense layer with the number of classes as its output.

- BatchNormalization Layer is introduced to speed up the training process and reduce chances of weight decay.
    * axis: The axis parameter determines the axis along which the normalization is applied. In this case, axis=-1 indicates that normalization is performed along the last axis, which typically corresponds to the features or channels in an image. By normalizing along this axis, each feature/channel is independently normalized.
    * momentum: The momentum parameter controls the update of the moving averages that are used during training and inference. A value of 0.99 implies that the moving averages are updated by taking 99% of the previous values and 1% of the new batch statistics in each training step. This helps in stabilizing the normalization process and reducing the impact of individual batch statistics.
    * epsilon: The epsilon parameter is a small value added to the denominator during normalization to avoid division by zero. It ensures numerical stability and prevents potential issues when the standard deviation is close to zero. In this case, epsilon=0.001 ensures that a small constant is added to the denominator for stability purposes.
- Dense Layer(D1) is introduced to improve the performance and generalization of the neural network.
    * Units: The dense layer has 256 units (neurons). This means that the layer will output a vector of size 256.
    * Kernel Regularization: The kernel_regularizer parameter is set to regularizers.l2(l=0.016). L2 regularization, also known as weight decay, is applied to the weights (kernels) of the layer. It helps to prevent overfitting by adding a penalty term to the loss function that encourages the weights to stay small.
    * Activity Regularization: The activity_regularizer parameter is set to regularizers.l1(0.006). L1 regularization is applied to the layer's activations. It adds a penalty to the loss function based on the absolute values of the activations, promoting sparsity and reducing the complexity of the model.
    * Bias Regularization: The bias_regularizer parameter is set to regularizers.l1(0.006). L1 regularization is applied to the biases of the layer. It helps to reduce the complexity of the model by encouraging some of the biases to be close to zero.
    * Activation Function: The activation function used in this dense layer is the Rectified Linear Unit (ReLU). ReLU applies an element-wise non-linearity, setting negative values to zero and leaving positive values unchanged.
- Dropout Layer(D2) is introduced with a rate of 0.45 and a seed of 123 is a regularization technique commonly used in deep learning models, including those used for image classification. Its purpose is to prevent overfitting
- Dense Layer(D2) is introduced with the activation function set to 'sigmoid' and the number of neurons equal to the class count is commonly used as the output layer in binary classification tasks.

### Training Backend 
- Tensorflow
- Keras
- GPU P100

### Other parameters used in the model
- Optimizer: Adamax
- Learning rate: 0.0019
- Loss: Binary Crossentropy
- Batch Size: 45
- Epochs: 100

### Additionally more parameters have been used for accurate training
- patience : number of epochs to wait to adjust lr if monitored value does not improve
- stop_patience : number of epochs to wait before stopping training if monitored value does not improve
- threshold : if train accuracy is < threshhold adjust monitor accuracy, else monitor validation loss
- factor : factor to reduce lr by
- ask_epoch : number of epochs to run before asking if you want to halt training
- batches : number of training batch to run per epoch

All these addittional parameters have been used for the callback function.

<p align="center">
  <img src="https://github.com/sudipg4112001/Image-Manipulation-Detection/assets/60208804/03b2954f-16f7-4c08-86f6-7cef44ecf1f9" alt="Image" width="260">
</p>
Throughout the entire training process, the following parameters are taken into account for each epoch:

- LR (Learning Rate)
- Next LR (Next Learning Rate)
- Monitor
- Percentage Improvement (% Improv)
- Duration

The user sets the initial learning rate, which is then adjusted based on the monitored parameter. Initially, accuracy is monitored until it reaches 97%, after which val_loss is monitored. If the improvement percentage in the monitored parameter saturates or worsens (monitored for up to 5 epochs), the learning rate is reduced by a factor of 0.3. If this trend continues for 10 epochs, the training is halted.

# Result

### Training Results

- loss: 0.1675 
- accuracy: 0.9695
  
### Validation Results

- loss: 0.8296
- accuracy: 0.7676

### Test Results

- loss: 1.0446
- accuracy: 0.7048

## Training and Validation Plot 

### Accuracy and Validation

![image](https://github.com/sudipg4112001/Image-Manipulation-Detection/assets/60208804/c56e471a-5ce4-43db-b425-1b16e58303c5)


### Confusion Matrix

<img src="https://github.com/sudipg4112001/Image-Manipulation-Detection/assets/60208804/49f1b750-6f95-48aa-a241-2be8b6f3a6f1" alt="Image" width="600" height="400">
