use std::f32::consts::FRAC_2_PI;

use ndarray::Array2;

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
  attention: Attention,
  network: Network,
}

impl Block {
  pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
    let x = x + self.attention.apply(x);
    let x = &x + self.network.apply(&x);

    x
  }
}

pub struct Attention {
  layer_norm: LayerNorm,
  /// Multiplied with input before self-attention.
  pre_self_attention: Weights,
  /// Projects back down after self-attention.
  projection: Weights,
}

impl Attention {
  pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
    x.clone()
    // let x = self.layer_norm.apply(&x);
  }
}

pub struct Network {
  layer_norm: LayerNorm,
  expand: Weights,
  collapse: Weights,
}

impl Network {
  pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
    let x = self.layer_norm.apply(x);
    let a = self.expand.linear(&x).mapv(gelu);

    self.collapse.linear(&a)
  }
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

fn gelu(x: f32) -> f32 {
  0.5 * x * (1.0 + f32::tanh(f32::sqrt(FRAC_2_PI) * (x + 0.044715 * x.powi(3))))
}

fn softmax(x: &Array2<f32>) -> Array2<f32> {
  let exp_x = (x - x.max_keep_dims()).mapv(f32::exp);

  &exp_x / exp_x.sum_keep_dims()
}
