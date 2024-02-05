mod encoder;
mod ext;
mod hyper_params;
mod params;
mod utils;

use std::{collections::HashMap, f32::consts::FRAC_2_PI, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use hyper_params::HyperParams;
use ndarray::{Array2, Axis};

use self::encoder::Encoder;
use crate::{ext::ArrayExt, params::LayerNorm};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(long)]
  model_dir: PathBuf,

  #[arg(long)]
  prompt: String,
  // #[arg(long)]
  // num_tokens: usize,
}

struct Distribution<T> {
  pmf: HashMap<T, f64>,
}

fn gelu(x: f32) -> f32 {
  0.5 * x * (1.0 + f32::tanh(f32::sqrt(FRAC_2_PI) * (x + 0.044715 * x.powi(3))))
}

fn softmax(x: &Array2<f32>) -> Array2<f32> {
  let exp_x = (x - x.max_keep_dims()).mapv(f32::exp);

  &exp_x / exp_x.sum_keep_dims()
}

// fn gpt(tokens: &[TokenId]) -> Result<Vec<Distribution<TokenId>>> {}

// fn generate(mut tokens: Vec<TokenId>, num_tokens: usize) -> Result<Vec<TokenId>> {
//   for i in 0..num_tokens {
//     let dist = gpt(&tokens).context("gpt")?.last().context("no distributions")?;
//     tokens.push(dist.sample().context("sample")?);
//   }

//   Ok(tokens[tokens.len() - num_tokens..].to_vec())
// }

fn main() -> Result<()> {
  let args = Args::parse();

  let hyper_params = HyperParams::from_dir(&args.model_dir).context("hyper params")?;
  // let params = Params::from_dir(&args.params_dir).context("params")?;
  let mut encoder = Encoder::from_dir(&args.model_dir).context("encoder")?;

  let token_ids = encoder.encode(&args.prompt).context("encode")?;
  let decoded = encoder.decode(&token_ids);

  let a = ndarray::array![[1., 2.], [3., 4.]];
  let b = ndarray::array![[5., 6.], [7., 8.]];
  let c = ndarray::array![[9., 10.], [11., 12.]];

  println!("softmax(a) = {}", softmax(&a));
  println!("gelu(a) = {}", a.mapv(gelu));

  // let output_tokens = generate(input_tokens, args.num_tokens).context("generate")?;

  // let output = decoder::decode(output_tokens).context("decode")?;

  Ok(())
}
