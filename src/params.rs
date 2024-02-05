use ndarray::{Array2, Axis};

use crate::ext::ArrayExt;

/// GPT-2 model parameters.
pub struct Params {
  position_encoding: Array2<f32>,
  token_encoding: Array2<f32>,
  layer_norm: LayerNorm,
  blocks: Vec<Block>,
}

/// Learned parameters for layer normalization.
pub struct LayerNorm {
  pub beta: Array2<f32>,
  pub gamma: Array2<f32>,
}

impl LayerNorm {
  pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
    let eps = 1e-5;

    let mean = x.mean_keep_dims();
    let var = x.var_keep_dims();
    let x = (x - mean) / (var + eps).mapv(f32::sqrt);

    &self.gamma * x + &self.beta
  }
}

/// The decoder portion of a transformer block.
///
/// This is made of two sublayers:
/// - a multi-head causal self-attention
/// - a position-wise feed forward neural network
pub struct Block {
  /// Layer normalization for attention.
  layer_norm_1: LayerNorm,
  attention: Attention,

  /// Layer normalization for the nerural network.
  layer_norm_2: LayerNorm,
  nn_weights: Weights,
}

pub struct Attention {
  /// Multiplied with input before self-attention.
  pre_self_attention: Weights,
  /// Projects back down after self-attention.
  projection: Weights,
}

/// Generic weights for linear multiplication with bias.
pub struct Weights {
  weights: Array2<f32>,
  bias: Array2<f32>,
}

impl Weights {
  pub fn linear(&self, x: &Array2<f32>) -> Array2<f32> {
    x.dot(&self.weights) + &self.bias
  }
}
