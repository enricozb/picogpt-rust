use std::{collections::HashMap, hash::Hash};

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
