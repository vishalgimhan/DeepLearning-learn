# DeepLearning-learn

**01. Artificial Intelligence Foundations - Neural Networks**

**- Data Preprocessing**: Essential libraries such as pandas, numpy, matplotlib, seaborn, and tensorflow are imported. The dataset is loaded into a pandas DataFrame from a CSV file. The dataset is explored using methods like info(), describe(), and checking for null values. Correlation among variables is analyzed using a heatmap to understand relationships between features and the target variable (sales).

**- Exploratory Data Analysis (EDA)**: Scatter plots are created to visualize the linear relationship between each feature and the target variable (sales). Heatmaps are used to show the correlation matrix and highlight the most related variables with sales.

**- Data Preprocessing**: The dataset is split into features (X) and target (y). Features are normalized using keras.utils.normalize. The dataset is split into training and testing sets using train_test_split.

**- Model Building**: A Sequential model is built using Keras with multiple Dense layers. The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function. The model is trained on the training data with validation on the test data for 32 epochs.

** -Model Evaluation**: The training and validation loss are plotted to evaluate the model's performance over epochs. The model's predictions on the test set are compared with the true values. The model is tuned by adding additional layers and neurons. The learning rate and batch size are adjusted to improve model performance. Early stopping is implemented to prevent overfitting.

**02. Deep Learning - Kaggle**

**- Deep Learning Fundamentals**: Understanding the basics of deep learning, characterized by deep stacks of computations that enable models to learn complex and hierarchical patterns.

**- Single Neuron**: Learning how individual neurons perform simple computations and how their connections form the power of neural networks.

**- Linear Units in Keras:** Defining and visualizing linear models using Keras, and understanding the role of weights and biases in neural networks.

**- Deep Neural Networks:** Organizing neurons into layers to perform complex data transformations, and understanding the importance of deep stacks of layers.

**- Activation Functions:** Introducing non-linearity into the model using activation functions like ReLU, which enable the network to learn complex patterns.

**- Stacking Dense Layers:** Building complex data transformations by stacking dense layers with activation functions, making the network suitable for various tasks.

**- Sequential Models:** Using the Sequential model to connect layers in order, from input to output, for straightforward model building.

**- Stochastic Gradient Descent (SGD):** Understanding the iterative algorithm that adjusts weights to minimize loss, using random samples from the dataset.

**- Loss Function:** Learning how the loss function measures the disparity between the target's true value and the model's prediction, and its importance in training.

**- Optimizer:** Using optimizers like Adam to adjust weights and minimize loss, and understanding their role in the training process.
