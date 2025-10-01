# Crystal Structure Prediction with a Neural Network(Part-1)

## Project Goal

The goal of this project is to develop a machine learning model capable of predicting the crystal structure distortion of perovskite compounds based on their chemical and physical properties. This is a multi-class classification task where the model predicts one of four distortion types: **cubic**, **orthorhombic**, **tetragonal**, or **rhombohedral**.



---

## Approach

The project was executed through a structured data science workflow, from data cleaning and preparation to model training and comparative evaluation.

**1. Data Loading and Cleaning**
* The dataset was loaded from `Crystal_structure.csv`, containing over 5,000 compounds with 15 distinct features.
* Initial cleaning involved removing irrelevant columns (`Compound`, `A`, `B`) and converting placeholder values (`-`) to `NaN` (Not a Number) for proper handling.
* Categorical features like `In literature` and the target variable `Lowest distortion` were numerically encoded.

**2. Handling Missing Data (Imputation)**
A key challenge was the presence of missing values in the dataset. A hybrid approach was used:
* **KNN Imputer:** For the features `τ` and `Lowest distortion`, missing values were imputed using the K-Nearest Neighbors algorithm (`n_neighbors=7`), which calculates a value based on the closest examples in the dataset.
* **Forward Fill:** For the `v(A)` and `v(B)` features, a forward-fill (`ffill`) method was used to propagate the last valid observation forward.

**3. Exploratory Data Analysis (EDA)**
A correlation matrix and heatmap were generated to understand the relationships between different features in the dataset. This helps identify which properties are most strongly related to each other.



**4. Model Development and Training**
* **Model Architecture:** A feed-forward neural network (also known as a Multi-Layer Perceptron) was built using TensorFlow/Keras. The architecture consists of:
    * An input layer and three hidden `Dense` layers with `relu` activation (512, 256, and 256 neurons, respectively).
    * `Dropout` layers with a rate of 0.2 were added after each hidden layer to prevent overfitting.
    * A final `Dense` output layer with 4 units and `softmax` activation for multi-class classification.
* **Optimizer Comparison:** To find the most effective training algorithm for this problem, the model was trained and evaluated separately with four different optimizers: **SGD**, **RMSprop**, **Adadelta**, and **Adam**.
* **Training Process:** The dataset was split into training (64%), validation (16%), and test (20%) sets. Each model variant was trained for **120 epochs**.

---

## Key Findings and Interpretation

The primary goal was to compare the effectiveness of different optimizers on this specific dataset. The final test accuracies are summarized below.

| Rank | Optimizer | Test Accuracy | Key Observations & Interpretation |
| :--- | :--- | :--- | :--- |
| 1 | **SGD** | **86.87%** | The Stochastic Gradient Descent optimizer achieved the **highest test accuracy**. It proved to be a very effective and stable choice for this tabular dataset. |
| 2 | **Adam** | **86.77%** | The Adam optimizer performed almost identically to SGD, delivering strong and reliable results. It is a popular and robust default choice for many deep learning tasks. |
| 3 | **RMSprop** | **86.30%** | RMSprop also performed very well, with its accuracy being very close to that of SGD and Adam, confirming its suitability for this problem. |
| 4 | **Adadelta** | **75.61%** | Adadelta was the clear underperformer in this comparison, achieving a significantly lower accuracy than the other three optimizers. This suggests its adaptive learning rate mechanism was less suited to this specific problem's loss landscape. |

---

## Conclusion

The project successfully developed a neural network to predict perovskite crystal structure distortions with high accuracy. The comparative analysis demonstrated that for this dataset and model architecture, the **SGD, Adam, and RMSprop optimizers all performed exceptionally well**, achieving around **86-87% accuracy** on the unseen test data.

The **Adadelta optimizer was found to be less effective**. Overall, the high accuracy achieved confirms that the chemical and physical properties in the dataset are strong predictors of the final crystal structure.

# COVID-19 Chest X-Ray Classification using a CNN(Part-2)

## Project Goal

The objective of this project is to build and train a Convolutional Neural Network (CNN) to accurately classify chest X-ray images into three distinct categories: **COVID-19**, **Normal**, and **Viral Pneumonia**.

## Approach

The methodology for this project involved a standard deep learning workflow for image classification.

**1. Data Preparation**
The dataset consists of 251 training images and 66 test/validation images, distributed across the three classes. We used the `ImageDataGenerator` from Keras for preprocessing:
* Pixel values were rescaled to a `[0, 1]` range.
* **Data augmentation** (rotation, shear, zoom, and horizontal flipping) was applied to the training set. This artificially expands the dataset, helping the model generalize better and prevent overfitting.

**2. Model Architecture**
A custom sequential CNN was constructed with the following key layers:
* Four `Conv2D` layers with increasing filter sizes (32, 64, 128, 128) and `relu` activation to extract features from the images.
* `MaxPooling2D` layers after each convolutional block to reduce spatial dimensions.
* A `Flatten` layer to prepare the features for the fully connected layers.
* A `Dense` layer with 512 units and a `Dropout` layer (rate of 0.5) to prevent overfitting.
* A final `Dense` output layer with 3 units and `softmax` activation to produce the class probabilities.

**3. Training and Evaluation**
* The model was compiled using the **Adam optimizer** and **categorical cross-entropy** loss function.
* Training was conducted for **20 epochs**, using the validation set to monitor performance on unseen data during the training process.
* The final trained model was evaluated on the test set to get an unbiased measure of its performance.

---

## Key Findings and Interpretation

**1. Final Performance**
The model achieved a final **test accuracy of 83.3%**, demonstrating a strong ability to distinguish between the different types of chest X-rays.

**2. Training Dynamics**
The model showed a successful learning progression.
* It quickly learned to differentiate the classes, with validation accuracy jumping from **~39% to over 90%** within the first 10 epochs.
* The performance on the validation set was somewhat unstable in the later epochs, fluctuating between 82% and 93%. This suggests the model is effective but might benefit from a larger dataset or further hyperparameter tuning to stabilize its learning.

**3. Training History Plots**
The accuracy and loss curves below visualize the model's performance over the 20 epochs.




**4. Single Image Prediction**
The model was also tested on a single, unseen image from the test set and successfully predicted its label as "Normal," confirming its practical applicability.

---

## Conclusion

This project successfully developed a custom CNN capable of classifying chest X-rays for COVID-19, Normal, and Viral Pneumonia cases with **83.3% accuracy**. The results are very promising and show the potential of deep learning for medical image analysis.

Future work could involve training on a larger dataset, experimenting with more complex architectures (like ResNet or VGGNet), and fine-tuning hyperparameters to further improve accuracy and stability.

# Comparative Analysis of CNNs for Flow Classification(Part-3):
## Project Goal:

To implement, train, and evaluate several classic CNN architectures to determine the most effective model for classifying fluid dynamics images as either 'laminar' or 'turbulent'.

## Approach:

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

## Key Findings and Interpretation:

Here is a summary of the findings for each model, ranked by performance on the test data.

### Model Performance Summary

| Rank | Model | Test Accuracy | Test Loss | Key Observations & Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **ZFNet** | **93.02%** | **0.15** | This was the **best-performing model**. It showed a strong ability to learn from the training data and generalized very well to the unseen test data. |
| 2 | **LeNet-5** | **88.37%** | **0.40** | Despite being a simple architecture, LeNet-5 performed remarkably well. It showed signs of **overfitting** but still generalized effectively. |
| 3 (tie) | **AlexNet** | **53.49%** | **0.69** | This model **failed to learn** effectively. The accuracy remained slightly better than random guessing, suggesting it was too complex for the small dataset or required more tuning. |
| 3 (tie) | **GoogLeNet**| **53.49%** | **0.69** | Similar to AlexNet, GoogLeNet also **failed to learn**. The accuracy was stuck at the baseline, indicating it could not find meaningful patterns in the data. |
| 5 (tie) | **VGGNet** | **46.51%** | **NaN** | This model **completely failed**. The loss became `NaN` (Not a Number) due to unstable training. Its performance is worse than random guessing. |
| 5 (tie) | **ResNet** | **46.51%** | **10.65** | This model also **failed to generalize**. While it achieved high training accuracy, it performed very poorly on the validation and test data, a classic case of **severe overfitting**. |

Based on the experiments, the **ZFNet** model was the best-performing architecture.

## Conclusion:

Our comparative analysis clearly demonstrates that for our dataset of laminar and turbulent flow images, the ZFNet architecture provided the best balance of complexity and learning capacity, achieving an excellent test accuracy of 93.02%.

Interestingly, the simpler LeNet-5 also proved highly effective, outperforming much larger and more complex models like AlexNet, GoogLeNet, VGGNet, and ResNet. This suggests that for the dataset size, the massive complexity of the more modern networks led to significant training difficulties and overfitting, while the more constrained architectures were better suited to the task.







