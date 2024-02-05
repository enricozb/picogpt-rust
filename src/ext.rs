use std::{collections::HashMap, hash::Hash};

use ndarray::{Array2, Axis};

#[extend::ext(name = HashMapExt)]
pub impl<K, V> HashMap<K, V> {
  fn invert(&self) -> HashMap<V, K>
  where
    K: Clone,
    V: Clone + Hash + Eq,
  {
    self.iter().map(|(k, v)| (v.clone(), k.clone())).collect()
  }
}

#[extend::ext(name = ArrayExt)]
pub impl Array2<f32> {
  /// Equivalent to `np.mean(x, -1, keep_dims=True)`
  fn mean_keep_dims(&self) -> Array2<f32> {
    let mean = self.mean_axis(Axis(1)).unwrap();
    let mean_shape = mean.shape()[0];
    mean.into_shape((mean_shape, 1)).unwrap()
  }

  /// Equivalent to `np.var(x, -1, keep_dims=True)`
  fn var_keep_dims(&self) -> Array2<f32> {
    let var = self.var_axis(Axis(1), /* ddof */ 0.0);
    let var_shape = var.shape()[0];
    var.into_shape((var_shape, 1)).unwrap()
  }
}
