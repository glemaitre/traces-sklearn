# %%
import numpy as np

n_samples = 30  # Play with me
n_features = 1

rng = np.random.default_rng(42)
X = rng.normal(size=(n_samples, n_features))
coef = np.ones(n_features) * 10

# linear relationship
y = X @ coef
# some noise
y += 5 *rng.normal(size=n_samples)

# %%
import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.show()

# %%
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# %%
from sklearn.model_selection import RepeatedKFold

cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)

for train_index, test_index in cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test, alpha=0.1)
    plt.plot(X_test, y_pred, color="tab:orange", linewidth=3, alpha=0.1)

# %%
