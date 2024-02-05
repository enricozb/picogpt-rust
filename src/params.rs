use std::{f32::consts::FRAC_2_PI, path::Path};

use anyhow::{Context, Result};
use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::{
  encoder::TokenId,
  ext::{ArrayView2Ext, VecExt},
};

/// GPT-2 model parameters.
pub struct Params {
  token_embedding: Array2<f32>,
  position_embedding: Array2<f32>,
  blocks: Vec<Block>,
  layer_norm: LayerNorm,
}

impl Params {
  pub fn from_dir<P: AsRef<Path>>(model_dir: P, num_heads: usize, depth: usize) -> Result<Self> {
    let model_dir = model_dir.as_ref();

    Ok(Self {
      token_embedding: ndarray_npy::read_npy(model_dir.join("wte.npy")).context("wte")?,
      position_embedding: ndarray_npy::read_npy(model_dir.join("wpe.npy")).context("wpe")?,
      blocks: Block::from_dirs(&model_dir, num_heads, depth).context("block")?,
      layer_norm: LayerNorm::from_dir(model_dir.join("ln_f")).context("layer norm")?,
    })
  }

  pub fn generate(&self, mut inputs: Vec<TokenId>, num_tokens: usize) -> Vec<TokenId> {
    for _ in 0..num_tokens {
      let logits = self.gpt2(&inputs);

      let next_token_id = logits
        .view()
        .slice_vec(&[logits.shape()[0] - 1])
        .unwrap()
        .index_axis(Axis(0), 0)
        .to_vec()
        .argmax()
        .unwrap();

      inputs.push(next_token_id as u64)
    }

    inputs[inputs.len() - num_tokens..].to_vec()
  }

  fn gpt2(&self, inputs: &[TokenId]) -> Array2<f32> {
    let inputs: Vec<usize> = inputs.iter().map(|token_id| *token_id as usize).collect();

    let token_embeddings = self.token_embedding.view().slice_vec(&inputs).unwrap();
    let position_embeddings = self.position_embedding.slice_axis(Axis(0), (0..inputs.len()).into());

    let mut x = token_embeddings + position_embeddings;

    for block in &self.blocks {
      x = block.apply(&x);
    }

    x = self.layer_norm.apply(&x);

    x.dot(&self.token_embedding.t())
  }
}

/// Learned parameters for layer normalization.
pub struct LayerNorm {
  pub beta: Array1<f32>,
  pub gamma: Array1<f32>,
}

impl LayerNorm {
  fn from_dir<P: AsRef<Path>>(layer_norm_dir: P) -> Result<Self> {
    let layer_norm_dir = layer_norm_dir.as_ref();

    Ok(Self {
      beta: ndarray_npy::read_npy(layer_norm_dir.join("b.npy"))?,
      gamma: ndarray_npy::read_npy(layer_norm_dir.join("g.npy"))?,
    })
  }

  pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
    let eps = 1e-5;

    let mean = x.view().mean_keep_dims();
    let var = x.view().var_keep_dims();
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
  fn from_dirs<P: AsRef<Path>>(model_dir: P, num_heads: usize, depth: usize) -> Result<Vec<Self>> {
    let model_dir = model_dir.as_ref();

    (0..depth)
      .map(|block_idx| Self::from_dir(model_dir.join(format!("h{block_idx}")), num_heads))
      .collect()
  }

  fn from_dir<P: AsRef<Path>>(block_dir: P, num_heads: usize) -> Result<Self> {
    let block_dir = block_dir.as_ref();

    Ok(Self {
      attention: Attention {
        layer_norm: LayerNorm::from_dir(block_dir.join("ln_1")).context("ln_1")?,
        pre_self_attention: Weights::from_dir(block_dir.join("attn/c_attn")).context("attn/c_attn")?,
        num_heads,
        post_self_attention: Weights::from_dir(block_dir.join("attn/c_proj")).context("attn/c_proj")?,
      },

      network: Network {
        layer_norm: LayerNorm::from_dir(block_dir.join("ln_2")).context("ln_2")?,
        expand: Weights::from_dir(block_dir.join("mlp/c_fc")).context("mlp/c_fc")?,
        collapse: Weights::from_dir(block_dir.join("mlp/c_proj")).context("mlp/c_proj")?,
      },
    })
  }

  pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
    let x = x + self.attention.apply(x);
    let x = &x + self.network.apply(&x);

    x
  }
}

pub struct Attention {
  /// Normalize input.
  layer_norm: LayerNorm,
  /// Multiplied with input before self-attention.
  pre_self_attention: Weights,
  /// Number of heads.
  num_heads: usize,
  /// Projects back down after self-attention.
  post_self_attention: Weights,
}

impl Attention {
  pub fn apply(&self, x: &Array2<f32>) -> Array2<f32> {
    let x = self.layer_norm.apply(x);

    let x = self.pre_self_attention.linear(&x);
    let x = x.view();
    let qkv = x.split_n(3).unwrap();
    let qkv_heads: Vec<_> = qkv.iter().map(|x| x.split_n(self.num_heads).unwrap()).collect();

    let causal_mask = (1.0 - tri(x.shape()[0])) * -1e10;

    let out_heads: Vec<_> = std::iter::zip(std::iter::zip(&qkv_heads[0], &qkv_heads[1]), &qkv_heads[2])
      .map(|((q, k), v)| attention(q, k, v, &causal_mask))
      .collect();

    let out_heads: Vec<_> = out_heads.iter().map(Array2::view).collect();
    let out_heads = ndarray::concatenate(Axis(1), &out_heads[..]).unwrap();

    self.post_self_attention.linear(&out_heads)
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
  bias: Array1<f32>,
}

impl Weights {
  fn from_dir<P: AsRef<Path>>(weights_dir: P) -> Result<Self> {
    let weights_dir = weights_dir.as_ref();

    Ok(Self {
      weights: ndarray_npy::read_npy(weights_dir.join("w.npy"))?,
      bias: ndarray_npy::read_npy(weights_dir.join("b.npy"))?,
    })
  }
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
  let exp_x = (x - x.view().max_keep_dims()).mapv(f32::exp);

  &exp_x / exp_x.view().sum_keep_dims()
}

fn attention(q: &ArrayView2<f32>, k: &ArrayView2<f32>, v: &ArrayView2<f32>, causal_mask: &Array2<f32>) -> Array2<f32> {
  softmax(&(q.dot(&k.t()) / (q.shape()[1] as f32).sqrt() + causal_mask)).dot(v)
}

pub fn tri(size: usize) -> Array2<f32> {
  let mut a = Array2::<f32>::zeros((size, size));

  for r in 0..size {
    for c in 0..=r {
      a[[r, c]] = 1.0;
    }
  }

  a
}
