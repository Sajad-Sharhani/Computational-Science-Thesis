import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Load the dataset
df = pd.read_csv('../data/adult.csv')

# Preprocessing
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('income-per-year')  # Target column
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

sensitive_attributes = ['race', 'sex']
permissible_attributes = list(set(categorical_cols + numerical_cols) - set(sensitive_attributes))
target = 'income-per-year'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
    ],
    sparse_threshold=0
)

X = df.drop(target, axis=1)
y = df[target].apply(lambda x: 1 if x == '>50K' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
sensitive_feature_indices = [i for i, col in enumerate(X_train.columns)
                             if col in sensitive_attributes]
permissible_feature_indices = [i for i, col in enumerate(X_train.columns)
                               if col not in sensitive_attributes]
X_test_sensitive = X_test_transformed[:, sensitive_feature_indices]
X_test_permissible = X_test_transformed[:, permissible_feature_indices]

# Assuming X_train_transformed is appropriately split into sensitive and permissible features
# This requires additional preprocessing not shown here

# Define model architecture with latent space
input_sensitive = tf.keras.Input(shape=(X_train_transformed.shape[1],), name='input_sensitive')
hidden1_sensitive = tf.keras.layers.Dense(10, activation='relu',
                                          name='hidden1_sensitive')(input_sensitive)
latent_sensitive = tf.keras.layers.Dense(5, activation='relu',
                                         name='latent_sensitive')(hidden1_sensitive)

input_permissible = tf.keras.Input(shape=(X_train_transformed.shape[1],), name='input_permissible')
hidden1_permissible = tf.keras.layers.Dense(10, activation='relu',
                                            name='hidden1_permissible')(input_permissible)
latent_permissible = tf.keras.layers.Dense(5, activation='relu',
                                           name='latent_permissible')(hidden1_permissible)

# Combine Outputs with Latent Space
combined = tf.keras.layers.concatenate([latent_sensitive, latent_permissible], name='combined')
hidden_combined = tf.keras.layers.Dense(10, activation='relu', name='hidden_combined')(combined)
final_output = tf.keras.layers.Dense(1, activation='sigmoid', name='final_output')(hidden_combined)

model = tf.keras.Model(inputs=[input_sensitive, input_permissible], outputs=final_output)


# Custom Loss Function
def custom_loss_wrapper(alpha=2.0, beta=5.0):
    def custom_loss(y_true, y_pred):
        accuracy_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        fairness_loss = tf.constant(0.9)  # Placeholder for demonstration
        conservatism_loss = tf.constant(0.7)  # Placeholder for demonstration
        total_loss = accuracy_loss + alpha * fairness_loss + beta * conservatism_loss
        return total_loss
    return custom_loss


# Compile the model with the custom loss
model.compile(optimizer='adam', loss=custom_loss_wrapper(alpha=1.0, beta=1.0), metrics=['accuracy'])

# Train the model
history = model.fit(
    # This should be split into sensitive and permissible parts
    [X_train_transformed, X_train_transformed],
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.3
)

# Evaluate the model
# Likewise, adjust for split inputs
eval_result = model.evaluate([X_test_transformed, X_test_transformed], y_test)
print("Model - Loss: {:.4f}, Accuracy: {:.4f}".format(eval_result[0], eval_result[1]))


def disparate_impact(y_pred, sensitive_column):
    """ Disparate Impact (DI) """
    di_num = np.mean(y_pred[sensitive_column == 1])
    di_denom = np.mean(y_pred[sensitive_column == 0])
    return di_num / di_denom if di_denom != 0 else np.nan


def conditional_value(y_pred, sensitive_column):
    """ Conditional Value (CV) """
    cv_1 = np.mean(y_pred[sensitive_column == 1])
    cv_0 = np.mean(y_pred[sensitive_column == 0])
    return 1 - cv_1 - cv_0


def group_conditioned_measures(y_pred, y_true, sensitive_column):
    """ s-Accuracy, s-TPR, s-TNR, s-BCR """
    accuracy = np.mean(y_pred[y_true == sensitive_column])
    tpr = np.mean(y_pred[(y_true == 1) & (sensitive_column == 1)])
    tnr = np.mean(y_pred[(y_true == 0) & (sensitive_column == 0)])
    bcr = (tpr + tnr) / 2
    return accuracy, tpr, tnr, bcr


def s_calibration_plus(y_pred, y_true, sensitive_column):
    """ s-Calibration+ """
    return np.mean(y_true[(y_pred == 1) & (sensitive_column == 1)])


def s_calibration_minus(y_pred, y_true, sensitive_column):
    """ s-Calibration- """
    return np.mean(y_true[(y_pred == 0) & (sensitive_column == 1)])


# calculate fairness measures
y_pred = (model.predict([X_test_transformed, X_test_transformed]) > 0.5).astype(int)
print(y_pred)
y_pred = np.round(y_pred).flatten()
y_true = y_test.values
sensitive_column = X_test_sensitive[:, 0]

print("Disparate Impact: {:.4f}".format(disparate_impact(y_pred, sensitive_column)))
print("Conditional Value: {:.4f}".format(conditional_value(y_pred, sensitive_column)))
print("s-Accuracy: {:.4f}".format(group_conditioned_measures(y_pred, y_true, sensitive_column)[0]))
print("s-TPR: {:.4f}".format(group_conditioned_measures(y_pred, y_true, sensitive_column)[1]))
print("s-TNR: {:.4f}".format(group_conditioned_measures(y_pred, y_true, sensitive_column)[2]))
print("s-BCR: {:.4f}".format(group_conditioned_measures(y_pred, y_true, sensitive_column)[3]))
print("s-Calibration+: {:.4f}".format(s_calibration_plus(y_pred, y_true, sensitive_column)))
print("s-Calibration-: {:.4f}".format(s_calibration_minus(y_pred, y_true, sensitive_column)))
