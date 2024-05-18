import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('../../data/adult.csv')

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

sensitive_features_train = X_train['sex']
sensitive_features_test = X_test['sex']

# Define and train the initial model
initial_model = LogisticRegression(solver='liblinear')
initial_model.fit(X_train_transformed, y_train)

# Predictions
y_pred_initial = initial_model.predict(X_test_transformed)

# Evaluate the initial model
metric_frame = MetricFrame(metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
                           y_true=y_test,
                           y_pred=y_pred_initial,
                           sensitive_features=sensitive_features_test)

print("Initial model metrics:")
print(metric_frame.overall)
print(metric_frame.by_group)

dpd_initial = demographic_parity_difference(y_test, y_pred_initial, sensitive_features=sensitive_features_test)
eod_initial = equalized_odds_difference(y_test, y_pred_initial, sensitive_features=sensitive_features_test)

print(f"Initial Model - Demographic Parity Difference: {dpd_initial}")
print(f"Initial Model - Equalized Odds Difference: {eod_initial}")

# Fairlearn - Mitigate bias using Exponentiated Gradient Reduction
mitigator = ExponentiatedGradient(LogisticRegression(solver='liblinear'), constraints=DemographicParity())
mitigator.fit(X_train_transformed, y_train, sensitive_features=sensitive_features_train)

# Mitigated predictions
y_pred_mitigated = mitigator.predict(X_test_transformed)

# Evaluate the mitigated model
metric_frame_mitigated = MetricFrame(metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
                                     y_true=y_test,
                                     y_pred=y_pred_mitigated,
                                     sensitive_features=sensitive_features_test)

print("Mitigated model metrics:")
print(metric_frame_mitigated.overall)
print(metric_frame_mitigated.by_group)

dpd_mitigated = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=sensitive_features_test)
eod_mitigated = equalized_odds_difference(y_test, y_pred_mitigated, sensitive_features=sensitive_features_test)

print(f"Mitigated Model - Demographic Parity Difference: {dpd_mitigated}")
print(f"Mitigated Model - Equalized Odds Difference: {eod_mitigated}")

# Plotting results
def plot_metrics(metric_frame, title):
    overall = metric_frame.overall
    by_group = metric_frame.by_group

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Overall metrics
    overall.plot(kind='bar', ax=ax[0])
    ax[0].set_title(f'Overall Metrics - {title}')
    ax[0].set_ylim([0, 1])

    # Metrics by group
    by_group.plot(kind='bar', ax=ax[1])
    ax[1].set_title(f'Metrics by Group - {title}')
    ax[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.show()

plot_metrics(metric_frame, "Initial Model")
plot_metrics(metric_frame_mitigated, "Mitigated Model")

