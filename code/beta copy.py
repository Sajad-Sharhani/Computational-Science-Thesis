import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../data/adult.csv')

# Preprocessing
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('income-per-year')  # Target column
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
    ],
    sparse_threshold=0
)

X = df.drop('income-per-year', axis=1)
y = df['income-per-year'].apply(lambda x: 1 if x == '>50K' else 0)
sensitive_attributes = ['race', 'sex']
permissible_attributes = list(set(categorical_cols + numerical_cols) - set(sensitive_attributes))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
sensitive_feature_indices = [i for i, col in enumerate(preprocessor.get_feature_names_out())
                             if any(s_attr in col for s_attr in sensitive_attributes)]
permissible_feature_indices = [i for i, col in enumerate(preprocessor.get_feature_names_out())
                               if not any(s_attr in col for s_attr in sensitive_attributes)]


# Define model architecture with separate inputs for sensitive and permissible attributes
def create_dual_input_model(input_shape_sensitive, input_shape_permissible):
    # Sensitive Path
    input_sensitive = tf.keras.Input(shape=(input_shape_sensitive,))
    hidden_sensitive = tf.keras.layers.Dense(10, activation='relu')(input_sensitive)
    latent_sensitive = tf.keras.layers.Dense(5, activation='relu')(hidden_sensitive)

    # Permissible Path
    input_permissible = tf.keras.Input(shape=(input_shape_permissible,))
    hidden_permissible = tf.keras.layers.Dense(10, activation='relu')(input_permissible)
    latent_permissible = tf.keras.layers.Dense(5, activation='relu')(hidden_permissible)

    # Combine and create final output
    combined = tf.keras.layers.concatenate([latent_sensitive, latent_permissible])
    hidden_combined = tf.keras.layers.Dense(10, activation='relu')(combined)
    final_output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_combined)

    model = tf.keras.Model(inputs=[input_sensitive, input_permissible], outputs=final_output)
    return model


# Custom loss function wrapper
def custom_loss_wrapper(alpha=1.0, beta=1.0, baseline_predictions=None, x_train_sensitive=None):
    def custom_loss(y_true, y_pred):
        # Squeeze y_pred to ensure it matches the shape of y_true
        y_pred_squeezed = tf.squeeze(y_pred)

        accuracy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred_squeezed)
        # print the size of x_train_sensitive
        # tf.print(tf.size(x_train_sensitive[:, 0] == 0))
        positive_preds = y_pred_squeezed[x_train_sensitive[:, 0] == 1]
        negative_preds = y_pred_squeezed[x_train_sensitive[:, 0] == 0]

        if tf.size(positive_preds) > 0 and tf.size(negative_preds) > 0:
            fairness_loss = tf.math.squared_difference(
                tf.reduce_mean(positive_preds), tf.reduce_mean(negative_preds))
        else:
            fairness_loss = 0.0

        if baseline_predictions is not None:
            y_pred_expanded = tf.expand_dims(y_pred_squeezed, -1)  #
            conservatism_loss = tf.keras.losses.binary_crossentropy(
                baseline_predictions, y_pred_expanded)
        else:
            conservatism_loss = tf.constant(0.0, dtype=tf.float32)

        total_loss = accuracy_loss + alpha * fairness_loss + beta * conservatism_loss
        return total_loss
    return custom_loss


# Adjust the shapes for sensitive and permissible inputs
input_shape_sensitive = len(sensitive_feature_indices)
input_shape_permissible = len(permissible_feature_indices)

# Create the model
model = create_dual_input_model(input_shape_sensitive, input_shape_permissible)

# Split the transformed training and test data
X_train_sensitive = X_train_transformed[:, sensitive_feature_indices]
X_train_permissible = X_train_transformed[:, permissible_feature_indices]
X_test_sensitive = X_test_transformed[:, sensitive_feature_indices]
X_test_permissible = X_test_transformed[:, permissible_feature_indices]


betas = np.linspace(0, 1, 10)  # Adjust the range and number of beta values as needed

# Initialize lists to store results
accuracy_losses = []
fairness_losses = []
accuracies = []
mean_predictions = []
conservatism_losses = []


def conditional_value(y_pred, sensitive_column):
    """ Conditional Value (CV) """
    cv_1 = np.mean(y_pred[sensitive_column == 1])
    cv_0 = np.mean(y_pred[sensitive_column == 0])
    return 1 - cv_1 - cv_0


baseline_predictions = np.mean(y_train)
for beta in betas:
    print(f"Training model with beta: {beta}")
    # Recompile the model with the current beta
    # model.compile(optimizer='adam',
    #               loss=custom_loss_wrapper(alpha=1, beta=beta, baseline_predictions=y_train),
    #               metrics=['accuracy'])

    optimizer = tf.keras.optimizers.Adam()
    epochs = 10
    batch_size = 32

    # Assuming your dataset is already shuffled if necessary
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train_sensitive, X_train_permissible, y_train))
    train_dataset = train_dataset.batch(batch_size)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss_avg = tf.keras.metrics.Mean()

        for batch, (x_sensitive, x_permissible, y) in enumerate(train_dataset):
            # print the size of x_sensitive, x_permissible, y
            with tf.GradientTape() as tape:
                y_pred = model([x_sensitive, x_permissible], training=True)  # Get model predictions
                loss = custom_loss_wrapper(alpha=1, beta=1,
                                           baseline_predictions=y_train,
                                           x_train_sensitive=x_sensitive)(y, y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss_avg.update_state(loss)
            if batch % 10 == 0:  # Adjust the frequency of logging to your preference
                print(f"Epoch {epoch+1}, Batch {batch}, Loss: {epoch_loss_avg.result().numpy()}")

        # End of epoch - you can add validation logic here if you wish
        print(f"Epoch {epoch+1} finished, Avg Loss: {epoch_loss_avg.result().numpy()}")

    # Make predictions
    y_pred = model.predict([X_test_sensitive, X_test_permissible]).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate losses and mean prediction
    accuracy_loss = np.mean(np.abs(y_test - y_pred))
    fairness_loss = conditional_value(y_pred_binary, X_test_sensitive[:, 0])
    mean_prediction = np.mean(y_pred)
    conservatism_loss = np.mean(np.square(baseline_predictions - y_pred))
    accuracy = np.mean(y_test == y_pred_binary)

    # Store results
    accuracy_losses.append(accuracy_loss)
    fairness_losses.append(fairness_loss)
    mean_predictions.append(mean_prediction)
    accuracies.append(accuracy)
    conservatism_losses.append(conservatism_loss)

# Now, plot the results
plt.figure(figsize=(12, 8))

# Plot for losses
plt.subplot(3, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(betas, accuracies, label='Accuracy Loss')
plt.plot(betas, fairness_losses, label='Fairness Loss')
plt.title('Losses as a Function of Beta')
plt.xlabel('Beta')
plt.ylabel('Loss')
plt.legend()

# conservatism loss vs beta
plt.subplot(3, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(betas, conservatism_losses, label='Conservatism Loss')
plt.title('Conservatism Loss as a Function of Beta')
plt.xlabel('Beta')
plt.ylabel('Conservatism Loss')
plt.legend()

# mean prediction vs beta
plt.subplot(3, 1, 3)  # 2 rows, 1 column, 3rd subplot
plt.plot(betas, mean_predictions, label='Mean Prediction')
plt.title('Mean Prediction as a Function of Beta')
plt.xlabel('Beta')
plt.ylabel('Mean Prediction')
plt.legend()


plt.tight_layout()  # Adjusts subplot params so that subplots fit into the figure area
plt.show()
