import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


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

X_train_sensitive = X_train_transformed[:, sensitive_feature_indices]
X_train_permissible = X_train_transformed[:, permissible_feature_indices]
X_test_sensitive = X_test_transformed[:, sensitive_feature_indices]
X_test_permissible = X_test_transformed[:, permissible_feature_indices]

# Neural Network 1 (Sensitive Attributes)
input_sensitive = tf.keras.Input(shape=(X_train_sensitive.shape[1],), name='input_sensitive')
hidden1_sensitive = tf.keras.layers.Dense(10, activation='relu',
                                          name='hidden1_sensitive')(input_sensitive)
output_sensitive = tf.keras.layers.Dense(5, activation='relu',
                                         name='output_sensitive')(hidden1_sensitive)

# Neural Network 2 (Permissible Attributes)
input_permissible = tf.keras.Input(shape=(X_train_permissible.shape[1],),
                                   name='input_permissible')
hidden1_permissible = tf.keras.layers.Dense(10, activation='relu',
                                            name='hidden1_permissible')(input_permissible)
output_permissible = tf.keras.layers.Dense(5, activation='relu',
                                           name='output_permissible')(hidden1_permissible)

# Combine Outputs
combined = tf.keras.layers.Add(name='combined')([output_sensitive, output_permissible])
final_output = tf.keras.layers.Dense(1, activation='sigmoid', name='final_output')(combined)

model = tf.keras.Model(inputs=[input_sensitive, input_permissible], outputs=final_output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    [X_train_sensitive, X_train_permissible],
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.3
)

original_eval = model.evaluate([X_test_sensitive, X_test_permissible], y_test)

# Creating a new model that only uses permissible attributes
input_permissible_only = tf.keras.Input(shape=(X_train_permissible.shape[1],),
                                        name='input_permissible_only')
hidden1_permissible_only = model.get_layer('hidden1_permissible')(input_permissible_only)
output_permissible_only = model.get_layer('output_permissible')(hidden1_permissible_only)
final_output_only = model.get_layer('final_output')(output_permissible_only)

model_permissible_only = tf.keras.Model(inputs=input_permissible_only, outputs=final_output_only)

# Compile the new model
model_permissible_only.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Now evaluate the new model on permissible attributes only
permissible_only_eval = model_permissible_only.evaluate(X_test_permissible, y_test)

print("Original Model - Loss: {:.4f}, Accuracy: {:.4f}".format(original_eval[0], original_eval[1]))
print("Permissible-only Model - Loss: {:.4f}, Accuracy: {:.4f}".format(permissible_only_eval[0],
                                                                       permissible_only_eval[1]))
