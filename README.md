# DeepLearning-learn

**Data Preprocessing**

**- Importing Libraries**: Essential libraries such as pandas, numpy, matplotlib, seaborn, and tensorflow are imported.

**- Loading Data**: The dataset is loaded into a pandas DataFrame from a CSV file.

**- Data Exploration**: The dataset is explored using methods like info(), describe(), and checking for null values.

**- Correlation Analysis**: Correlation among variables is analyzed using a heatmap to understand relationships between features and the target variable (sales).

**Exploratory Data Analysis (EDA)**

**- Scatter Plots**: Scatter plots are created to visualize the linear relationship between each feature and the target variable (sales).

**- Heatmaps**: Heatmaps are used to show the correlation matrix and highlight the most related variables with sales.

**Data Preprocessing**

**- Feature Selection**: The dataset is split into features (X) and target (y).

**- Normalization**: Features are normalized using keras.utils.normalize.

**- Train-Test Split**: The dataset is split into training and testing sets using train_test_split.

**Model Building**

**- Model Architecture**: A Sequential model is built using Keras with multiple Dense layers.

**- Compilation**: The model is compiled with the Adam optimizer and Mean Squared Error (MSE) loss function.

**- Training**: The model is trained on the training data with validation on the test data for 32 epochs.

**Model Evaluation**

**- Loss Visualization**: The training and validation loss are plotted to evaluate the model's performance over epochs.

**- Predictions**: The model's predictions on the test set are compared with the true values.

**Hyperparameter Tuning**

**- Additional Layers and Neurons**: The model is tuned by adding additional layers and neurons.

**- Learning Rate and Batch Size**: The learning rate and batch size are adjusted to improve model performance.

**- Early Stopping**: Early stopping is implemented to prevent overfitting.
