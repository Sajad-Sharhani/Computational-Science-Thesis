import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Load the dataset
df = pd.read_csv('../data/adult.csv')

# Preprocessing
# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('income-per-year')  # Target column
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Define sensitive attributes and permissible attributes
sensitive_attributes = ['race', 'sex']
permissible_attributes = list(set(categorical_cols + numerical_cols) - set(sensitive_attributes))
target = 'income-per-year'

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)  # Ensure dense output
    ],
    sparse_threshold=0  # Ensure the output is a dense array
)


# Split the dataset
X = df.drop(target, axis=1)
y = df[target].apply(lambda x: 1 if x == '>50K' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply transformations
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


# Split into sensitive and permissible data
sensitive_feature_indices = [i for i, col in enumerate(X_train.columns)
                             if col in sensitive_attributes]
permissible_feature_indices = [i for i, col in enumerate(X_train.columns)
                               if col not in sensitive_attributes]

X_train_sensitive = X_train_transformed[:, sensitive_feature_indices]
X_train_permissible = X_train_transformed[:, permissible_feature_indices]
X_test_sensitive = X_test_transformed[:, sensitive_feature_indices]
X_test_permissible = X_test_transformed[:, permissible_feature_indices]

# Neural Network 1 (Sensitive Attributes)
input_sensitive = tf.keras.Input(shape=(X_train_sensitive.shape[1],))
hidden1_sensitive = tf.keras.layers.Dense(10, activation='relu')(input_sensitive)
output_sensitive = tf.keras.layers.Dense(5, activation='relu')(hidden1_sensitive)

# Neural Network 2 (Permissible Attributes)
input_permissible = tf.keras.Input(shape=(X_train_permissible.shape[1],))
hidden1_permissible = tf.keras.layers.Dense(10, activation='relu')(input_permissible)
output_permissible = tf.keras.layers.Dense(5, activation='relu')(hidden1_permissible)

# Combine Outputs
combined = tf.keras.layers.Add()([output_sensitive, output_permissible])
final_output = tf.keras.layers.Dense(1, activation='sigmoid')(combined)

# Model
model = tf.keras.Model(inputs=[input_sensitive, input_permissible], outputs=final_output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(
    [X_train_sensitive, X_train_permissible],
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.3  # for example, 20% of the training data used for validation
)


# Evaluate
model.evaluate([X_test_sensitive, X_test_permissible], y_test)

# Generate predictions
y_pred = model.predict([X_test_sensitive, X_test_permissible])
y_pred_classes = (y_pred > 0.5).astype("int32")


# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Assuming history = model.fit(...)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
