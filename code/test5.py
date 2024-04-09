import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score

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
sensitive_feature_indices = [i for i, col in enumerate(X_train.columns)
                             if col in sensitive_attributes]
permissible_feature_indices = [i for i, col in enumerate(X_train.columns)
                               if col not in sensitive_attributes]
X_test_sensitive = X_test_transformed[:, sensitive_feature_indices]
X_test_permissible = X_test_transformed[:, permissible_feature_indices]


# Define model architecture with latent spaces for sensitive and permissible attributes
def create_model(input_shape):
    input_layer = tf.keras.Input(shape=(input_shape,))
    # Split into sensitive and permissible paths
    # For simplicity, this example uses a single input; split your real data accordingly
    hidden_sensitive = tf.keras.layers.Dense(10, activation='relu')(input_layer)
    latent_sensitive = tf.keras.layers.Dense(5, activation='relu')(hidden_sensitive)

    hidden_permissible = tf.keras.layers.Dense(10, activation='relu')(input_layer)
    latent_permissible = tf.keras.layers.Dense(5, activation='relu')(hidden_permissible)

    # Combine and create final output
    combined = tf.keras.layers.concatenate([latent_sensitive, latent_permissible])
    hidden_combined = tf.keras.layers.Dense(10, activation='relu')(combined)
    final_output = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_combined)

    model = tf.keras.Model(inputs=input_layer, outputs=final_output)
    return model


# Custom loss function wrapper
def custom_loss_wrapper(alpha=1.0, beta=1.0, baseline_predictions=None):
    """
    Custom loss wrapper to include accuracy, fairness, and conservatism.

    Parameters:
    - alpha: Weight for the fairness loss component.
    - beta: Weight for the conservatism loss component.
    - baseline_predictions: Numpy array or Tensor of predictions based on biased decision making.
                            This is used to calculate conservatism loss.
    """
    def custom_loss(y_true, y_pred):
        # Basic accuracy loss (to minimize)
        accuracy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Fairness loss (to maximize, hence subtracted)
        # For simplicity, assuming y_pred and y_true are 1D arrays and fairness_loss
        # is a placeholder
        # Replace this with a proper fairness metric calculation
        # Example: Difference in positive prediction rates between groups
        fairness_loss = tf.abs(tf.reduce_mean(y_pred[y_true == 1]) - tf.reduce_mean(y_pred[y_true == 0]))

        # Conservatism loss (to minimize, how far predictions are from original biased decisions)
        # Assuming baseline_predictions is provided and has the same shape as y_pred
        if baseline_predictions is not None:
            conservatism_loss = tf.keras.losses.mean_squared_error(baseline_predictions, y_pred)
        else:
            # Default to 0 if no baseline predictions are provided
            conservatism_loss = tf.constant(0.0, dtype=tf.float32)

        # Total loss calculation
        total_loss = accuracy_loss + alpha * fairness_loss + beta * conservatism_loss
        return total_loss

    return custom_loss


# # Hyperparameter tuning setup
# alpha_values = [0.1, 0.5, 1.0, 2.0]
# beta_values = [0.1, 0.5, 1.0, 2.0]

# best_score = float('inf')
# best_alpha = None
# best_beta = None

# for alpha in alpha_values:
#     for beta in beta_values:
#         # Re-create the model to reset weights
#         model = create_model(X_train_transformed.shape[1])
#         model.compile(optimizer='adam', loss=custom_loss_wrapper(alpha=alpha, beta=beta),
#                       metrics=['accuracy'])

#         # Train the model using the training set
#         model.fit(X_train_transformed, y_train, epochs=5, batch_size=32, verbose=0)

#         # Evaluate the model using the validation set
#         predictions = model.predict(X_test_transformed)
#         score = accuracy_score(y_test, predictions.round())

#         # Update best hyperparameters based on accuracy
#         if score < best_score:
#             best_score = score
#             best_alpha = alpha
#             best_beta = beta

# print(f"Best Alpha: {best_alpha}, Best Beta: {best_beta}, Best Score: {best_score}")

# Final model training with best hyperparameters
model = create_model(X_train_transformed.shape[1])


def conditional_value(y_pred, sensitive_column):
    """ Conditional Value (CV) """
    cv_1 = np.mean(y_pred[sensitive_column == 1])
    cv_0 = np.mean(y_pred[sensitive_column == 0])
    return 1 - cv_1 - cv_0


# alpha values should be between 0 and 1
alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Example alpha values
conservatism_scores = []  # To store conservatism scores
fairness_scores = []  # To store fairness scores
mutual_information_scores = []  # To store mutual information scores
# Assuming you have the rest of your setup as previously described
for alpha in alpha_values:
    # Assuming your model compilation and fitting here as before
    model.compile(optimizer='adam',
                  loss=custom_loss_wrapper(alpha=alpha, beta=0.5, baseline_predictions=y_train),
                  metrics=['accuracy'])
    history = model.fit(X_train_transformed, y_train, epochs=10,
                        batch_size=32, validation_split=0.2, verbose=0)
    # Make predictions with the current model
    y_pred = model.predict(X_test_transformed).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Assuming X_test_sensitive is a numpy array with the sensitive column you're interested in
    # For the purpose of demonstration, let's say it's the first column
    sensitive_column = X_test_sensitive[:, 0]  # Update this as per your sensitive attribute index

    # Calculate fairness using conditional value
    fairness_score = conditional_value(y_pred_binary, sensitive_column)

    # Conservatism score and mutual information score calculations
    # Placeholder: Replace with your actual calculations
    conservatism_score = alpha  # Placeholder
    mutual_information_score = alpha / 2  # Placeholder

    # Append calculated scores to their respective lists
    conservatism_scores.append(conservatism_score)
    fairness_scores.append(fairness_score)
    mutual_information_scores.append(mutual_information_score)


# def disparate_impact(y_pred, sensitive_column):
#     """ Disparate Impact (DI) """
#     di_num = np.mean(y_pred[sensitive_column == 1])
#     di_denom = np.mean(y_pred[sensitive_column == 0])
#     return di_num / di_denom if di_denom != 0 else np.nan


# def conditional_value(y_pred, sensitive_column):
#     """ Conditional Value (CV) """
#     cv_1 = np.mean(y_pred[sensitive_column == 1])
#     cv_0 = np.mean(y_pred[sensitive_column == 0])
#     return 1 - cv_1 - cv_0


# def group_conditioned_measures(y_pred, y_true, sensitive_column):
#     """ s-Accuracy, s-TPR, s-TNR, s-BCR """
#     accuracy = np.mean(y_pred[y_true == sensitive_column])
#     tpr = np.mean(y_pred[(y_true == 1) & (sensitive_column == 1)])
#     tnr = np.mean(y_pred[(y_true == 0) & (sensitive_column == 0)])
#     bcr = (tpr + tnr) / 2
#     return accuracy, tpr, tnr, bcr


# def s_calibration_plus(y_pred, y_true, sensitive_column):
#     """ s-Calibration+ """
#     return np.mean(y_true[(y_pred == 1) & (sensitive_column == 1)])


# def s_calibration_minus(y_pred, y_true, sensitive_column):
#     """ s-Calibration- """
#     return np.mean(y_true[(y_pred == 0) & (sensitive_column == 1)])


# calculate fairness measures
# Correct call to model.predict
# y_pred = (model.predict(X_test_transformed) > 0.5).astype(int)
# print(y_pred)
# y_pred = np.round(y_pred).flatten()
# y_true = y_test.values
# sensitive_column = X_test_sensitive[:, 0]

# print("Disparate Impact: {:.4f}".format(disparate_impact(y_pred, sensitive_column)))
# print("Conditional Value: {:.4f}".format(conditional_value(y_pred, sensitive_column)))
# print("s-Accuracy: {:.4f}".format(group_conditioned_measures(y_pred, y_true,
# sensitive_column)[0]))
# print("s-TPR: {:.4f}".format(group_conditioned_measures(y_pred, y_true, sensitive_column)[1]))
# print("s-TNR: {:.4f}".format(group_conditioned_measures(y_pred, y_true, sensitive_column)[2]))
# print("s-BCR: {:.4f}".format(group_conditioned_measures(y_pred, y_true, sensitive_column)[3]))
# print("s-Calibration+: {:.4f}".format(s_calibration_plus(y_pred, y_true, sensitive_column)))
# print("s-Calibration-: {:.4f}".format(s_calibration_minus(y_pred, y_true, sensitive_column)))
# Plotting the results
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Conservatism vs. Alpha
axs[0].plot(alpha_values, conservatism_scores, marker='o', linestyle='-', color='blue')
axs[0].set_title('Impact of Conservatism (Alpha) on Conservatism Term')
axs[0].set_xlabel('Alpha')
axs[0].set_ylabel('Conservatism Term')

# Fairness vs. Alpha
axs[1].plot(alpha_values, fairness_scores, marker='o', linestyle='-', color='red')
axs[1].set_title('Impact of Conservatism (Alpha) on Fairness')
axs[1].set_xlabel('Alpha')
axs[1].set_ylabel('Fairness')

# Mutual Information vs. Alpha
axs[2].plot(alpha_values, mutual_information_scores, marker='o', linestyle='-', color='green')
axs[2].set_title('Dependence of Mutual Information on Alpha')
axs[2].set_xlabel('Alpha')
axs[2].set_ylabel('Mutual Information')

plt.tight_layout()
plt.show()
