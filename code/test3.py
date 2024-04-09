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

# Neural Network 1 (Sensitive Attributes)
input_sensitive = tf.keras.Input(shape=(X_train_transformed.shape[1],), name='input_sensitive')
hidden1_sensitive = tf.keras.layers.Dense(10, activation='relu',
                                          name='hidden1_sensitive')(input_sensitive)
latent_sensitive = tf.keras.layers.Dense(5, activation='relu',
                                         name='latent_sensitive')(hidden1_sensitive)

# Neural Network 2 (Permissible Attributes)
input_permissible = tf.keras.Input(shape=(X_train_transformed.shape[1],), name='input_permissible')
hidden1_permissible = tf.keras.layers.Dense(10, activation='relu',
                                            name='hidden1_permissible')(input_permissible)
latent_permissible = tf.keras.layers.Dense(5, activation='relu',
                                           name='latent_permissible')(hidden1_permissible)

# Combine Outputs
combined = tf.keras.layers.concatenate([latent_sensitive, latent_permissible], name='combined')
hidden_combined = tf.keras.layers.Dense(10, activation='relu', name='hidden_combined')(combined)
final_output = tf.keras.layers.Dense(1, activation='sigmoid', name='final_output')(hidden_combined)

model = tf.keras.Model(inputs=[input_sensitive, input_permissible], outputs=final_output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    [X_train_transformed, X_train_transformed],
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.3
)

eval_result = model.evaluate([X_test_transformed, X_test_transformed], y_test)

print("Model - Loss: {:.4f}, Accuracy: {:.4f}".format(eval_result[0], eval_result[1]))
