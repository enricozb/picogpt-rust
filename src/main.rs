mod encoder;

use std::collections::HashMap;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(long)]
  prompt: String,

  #[arg(long)]
  num_tokens: usize,
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

  // let input_tokens = encoder::encode(args.prompt).context("encode")?;
  // let output_tokens = generate(input_tokens, args.num_tokens).context("generate")?;

  // let output = decoder::decode(output_tokens).context("decode")?;

  Ok(())
}
