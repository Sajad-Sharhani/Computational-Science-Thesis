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
def custom_loss_wrapper(alpha=1.0, beta=1.0, baseline_predictions=None):
    def custom_loss(y_true, y_pred):
        accuracy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        fairness_loss = tf.abs(
            tf.reduce_mean(y_pred[y_true == 1]) - tf.reduce_mean(y_pred[y_true == 0]))
        if baseline_predictions is not None:
            conservatism_loss = tf.keras.losses.mean_squared_error(baseline_predictions, y_pred)
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

# Compile and fit the model
model.compile(optimizer='adam',
              loss=custom_loss_wrapper(alpha=1, beta=1, baseline_predictions=y_train),
              metrics=['accuracy'])

history = model.fit([X_train_sensitive, X_train_permissible],
                    y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions
y_pred = model.predict([X_test_sensitive, X_test_permissible]).flatten()
y_pred_binary = (y_pred > 0.5).astype(int)

# Assuming X_test_sensitive is a numpy array with the sensitive column you're interested in
# For the purpose of demonstration, let's say it's the first column
sensitive_column = X_test_sensitive[:, 0]  # Update this as per your sensitive attribute index


def conditional_value(y_pred, sensitive_column):
    """ Conditional Value (CV) """
    cv_1 = np.mean(y_pred[sensitive_column == 1])
    cv_0 = np.mean(y_pred[sensitive_column == 0])
    return 1 - cv_1 - cv_0


# Calculate fairness using conditional value
fairness_score = conditional_value(y_pred_binary, sensitive_column)

print(f"Fairness score: {fairness_score}")

# Accessing the history data
epochs = range(1, len(history.history['loss']) + 1)
loss = history.history['loss']
val_loss = history.history['val_loss']
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plotting total loss
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo-', label='Training loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'bo-', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
