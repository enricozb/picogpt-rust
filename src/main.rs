mod encoder;
mod ext;
mod hyper_params;
mod params;
mod utils;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use hyper_params::HyperParams;

use self::{encoder::Encoder, params::Params};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(long)]
  model_dir: PathBuf,

  #[arg(long)]
  prompt: String,

  #[arg(long)]
  num_tokens: usize,
}

fn main() -> Result<()> {
  let args = Args::parse();

  let hyper_params = HyperParams::from_dir(&args.model_dir).context("hyper params")?;
  let params = Params::from_dir(
    args.model_dir.join("exploded_model"),
    hyper_params.num_heads,
    hyper_params.network_depth,
  )
  .context("params")?;

  let mut encoder = Encoder::from_dir(&args.model_dir).context("encoder")?;

  let token_ids = encoder.encode(&args.prompt).context("encode")?;
  anyhow::ensure!(token_ids.len() < hyper_params.max_context, "input too large");

  let output_ids = params.generate(token_ids, args.num_tokens);
  let decoded = encoder.decode(&output_ids);

  println!("output = {decoded:?}");

  Ok(())
}
