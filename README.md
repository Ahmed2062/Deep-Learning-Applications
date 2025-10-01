

# Comparative Analysis of CNNs for Flow Classification(Part-3):
Project Goal: To implement, train, and evaluate several classic CNN architectures to determine the most effective model for classifying fluid dynamics images as either 'laminar' or 'turbulent'.
# Approach:

The methodology follows a systematic and comprehensive process for tackling a binary image classification problem (laminar vs. turbulent flow).

Data Preparation: We began by programmatically splitting our image dataset into three distinct sets: training (70%), validation (15%), and testing (15%). We used ImageDataGenerator for data preprocessing, which rescales pixel values and applies data augmentation (rotation, shifting, flipping) to the training set to create a more robust model.

Comparative Model Analysis: Instead of relying on a single architecture, we implemented and tested a wide array of historically significant Convolutional Neural Network (CNN) models. This is a strong comparative approach that allows you to see how different architectures perform on a specific dataset. The models we evaluated include:

LeNet-5 (1998): A pioneering, simple CNN.

AlexNet (2012): The deep learning model that revolutionized image classification.

ZFNet (2013): An improvement on AlexNet.

GoogLeNet / Inception (2014): Introduced the "inception module" for computational efficiency.

VGGNet (2014): Known for its simple and deep architecture using small 3x3 convolution filters.

ResNet (2015): Introduced "residual blocks" or skip connections to train extremely deep networks effectively.

Training and Evaluation: For each model, we followed a consistent workflow:

Compiled the model using the Adam optimizer and categorical cross-entropy loss function.

Trained the model for a set number of epochs, using the validation set to monitor performance on unseen data.

Evaluated the final trained model on the completely separate test set to get an unbiased measure of its final accuracy.

# Key Findings and Interpretation:

Here’s a summary of what the results from our experiments show, ranked from best to worst performing model on the test data.

<img width="794" height="617" alt="image" src="https://github.com/user-attachments/assets/ed7d60e9-730e-4ae8-ab4a-4ecc0cb4262b" />


# Conclusion:

Our comparative analysis clearly demonstrates that for our dataset of laminar and turbulent flow images, the ZFNet architecture provided the best balance of complexity and learning capacity, achieving an excellent test accuracy of 93.02%.

Interestingly, the simpler LeNet-5 also proved highly effective, outperforming much larger and more complex models like AlexNet, GoogLeNet, VGGNet, and ResNet. This suggests that for the dataset size, the massive complexity of the more modern networks led to significant training difficulties and overfitting, while the more constrained architectures were better suited to the task.







