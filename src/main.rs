mod encoder;
mod ext;
mod utils;

use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;

use self::encoder::Encoder;

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

  let mut encoder = Encoder::new(args.model_dir).context("encoder")?;

  let token_ids = encoder.encode(&args.prompt).context("encode")?;

  let decoded = encoder.decode(&token_ids);

  // let output_tokens = generate(input_tokens, args.num_tokens).context("generate")?;

  // let output = decoder::decode(output_tokens).context("decode")?;

  Ok(())
}
