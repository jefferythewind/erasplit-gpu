# ErasplitGBDT

**ErasplitGBDT** is a compact and efficient implementation of a GPU-accelerated gradient boosted decision tree (GBDT) model written from scratch in under 240 lines of pure Python.

This project is inspired by the structure of modern gradient boosting libraries like LightGBM, but redesigned for interpretability and experimentation. It emphasizes vectorized GPU execution with PyTorch and provides a clean, easy-to-extend codebase.

## Highlights

- **Fully GPU-native**: Uses PyTorch tensors for data representation and processing. Binning, histogram construction, and tree traversal all run on the GPU.
- **Histogram-based learning**: Implements fast histogram construction with `index_add_` for gradient and hessian accumulation.
- **Quantile binning**: Feature binning based on quantiles for better handling of outliers and consistent tree splits.
- **Era-aware architecture**: Designed for time-series data with grouped `era_id` structure.
- **Compact tree growth**: Recursively grows trees using greedy gain-based splitting, stopping at `max_depth`.
- **No dependencies beyond PyTorch, NumPy, scikit-learn**: Easy to install and use anywhere.

## Usage
```python
from erasplit import ErasplitGBDT
model = ErasplitGBDT(num_bins=10, max_depth=3, learning_rate=0.1, n_estimators=100)
model.fit(X_train, y_train, era_id)
y_pred = model.predict(X_test)
```

## Example
The repo includes example visualizations of decision boundaries compared to LightGBM, showing similar split behavior and faster training on small datasets.

## Roadmap
- Add support for categorical features
- Implement feature importance scores
- Optimize `predict_from_tree_batch` with GPU-accelerated tree traversal
- Add support for custom loss functions

---

Questions or ideas? Open an issue or fork the project to experiment with new GBDT ideas.
