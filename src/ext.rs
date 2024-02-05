use std::{collections::HashMap, hash::Hash};

use anyhow::{Context, Result};
use ndarray::{Array, Array1, Array2, ArrayView1, Axis, RemoveAxis};

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
  fn slice_vec(&self, indices: &[usize]) -> Result<Array2<f32>> {
    let view: Vec<ArrayView1<_>> = indices.iter().map(|idx| self.index_axis(Axis(0), *idx)).collect();

    ndarray::stack(Axis(0), &view).context("stack")
  }

  /// Equivalent to `np.max(x, -1, keep_dims=True)`
  fn max_keep_dims(&self) -> Array2<f32> {
    let mut res: Array1<f32> = Array::zeros(self.raw_dim().remove_axis(Axis(1)));

    for subview in self.axis_iter(Axis(1)) {
      for i in 0..res.shape()[0] {
        res[i] = f32::max(res[i], subview[i])
      }
    }

    // TODO: the above loop can be written so as to avoid this reshape
    let res_shape = res.shape()[0];
    res.into_shape((res_shape, 1)).unwrap()
  }

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

  /// Equivalent to `np.sum(x, -1, keep_dims=True)`
  fn sum_keep_dims(&self) -> Array2<f32> {
    let sum = self.sum_axis(Axis(1));
    let sum_shape = sum.shape()[0];
    sum.into_shape((sum_shape, 1)).unwrap()
  }
}
