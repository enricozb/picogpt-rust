use std::{
  collections::{HashMap, HashSet},
  path::Path,
};

use anyhow::{Context, Result};
use fancy_regex::Regex;

use crate::ext::HashMapExt;

pub type Token = String;
pub type TokenId = u64;

pub struct Encoder {
  /// Matches "words" on whitespace, contracition, or numeric boundaries.
  word_re: Regex,
  /// Maps arbitrary bytes to printable "latent" characters.
  byte_to_char: HashMap<u8, char>,
  /// Maps "latent" characters back to their original bytes.
  char_to_byte: HashMap<char, u8>,
  /// Maps bpe split strings to token ids.
  encoder: HashMap<Token, TokenId>,
  /// Inverse of `encoder`.
  decoder: HashMap<TokenId, Token>,
  /// Priority of BPE merges.
  bpe_ranks: HashMap<(String, String), usize>,
  /// BPE split tokenize_cache.
  tokenize_cache: HashMap<String, Vec<TokenId>>,
}

impl Encoder {
  pub fn from_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
    let model_dir = model_dir.as_ref();

    let byte_to_char = Self::byte_to_char();
    let encoder: HashMap<Token, TokenId> = crate::utils::serde_json_from_path(model_dir.join("encoder.json"))?;

    Ok(Self {
      word_re: Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap(),
      char_to_byte: byte_to_char.invert(),
      byte_to_char,
      decoder: encoder.invert(),
      encoder,
      bpe_ranks: crate::utils::bpe_ranks_from_path(model_dir.join("vocab.bpe"))?,
      tokenize_cache: HashMap::new(),
    })
  }

  pub fn encode(&mut self, text: &str) -> Result<Vec<TokenId>> {
    let mut token_ids = Vec::new();

    for word in self.word_re.find_iter(text) {
      let word: String = word
        .context("match")?
        .as_str()
        .as_bytes()
        .iter()
        .map(|b| self.byte_to_char.get(b).unwrap())
        .collect();

      if let Some(cached_token_ids) = self.tokenize_cache.get(&word) {
        token_ids.extend(cached_token_ids.iter());
      } else {
        let tokens = self.tokenize(word.clone());
        let new_token_ids: Vec<TokenId> = tokens
          .into_iter()
          .map(|token| {
            *self
              .encoder
              .get(&token)
              .with_context(|| format!("unexpected token {token}"))
              .unwrap()
          })
          .collect();

        token_ids.extend(new_token_ids.iter());

        self.tokenize_cache.insert(word, new_token_ids);
      }
    }

    Ok(token_ids)
  }

  pub fn decode(&self, token_ids: &[TokenId]) -> String {
    let tokens: String = token_ids
      .iter()
      .map(|token_id| {
        self
          .decoder
          .get(token_id)
          .with_context(|| format!("unexpected token id: {token_id}"))
          .unwrap()
          .as_str()
      })
      .collect::<Vec<_>>()
      .join("");

    String::from_utf8(tokens.chars().map(|c| *self.char_to_byte.get(&c).unwrap()).collect()).unwrap()
  }

  /// Tokenizes a string into into it's BPE strings. Splits on all characters
  /// then performs all BPE merges.
  ///
  /// For example, "elephant" -> ["ele", "phant"]
  fn tokenize(&self, word: String) -> Vec<Token> {
    let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();

    loop {
      let pairs: HashSet<(String, String)> = Self::parts_to_pairs(&parts);

      // grab the highest priority merge
      let Some(bigram) = pairs
        .iter()
        .min_by_key(|pair| self.bpe_ranks.get(pair).copied().unwrap_or(usize::MAX))
      else {
        return vec![word];
      };

      if !self.bpe_ranks.contains_key(bigram) {
        // no more bpe merges
        break;
      }

      let (first, second) = bigram;

      let mut new_parts = Vec::new();
      let mut i = 0;
      while i < parts.len() {
        if i == parts.len() - 1 {
          new_parts.push(parts[i].to_string());

          break;
        }

        if &parts[i] == first && &parts[i + 1] == second {
          new_parts.push(format!("{first}{second}"));

          i += 2;
        } else {
          new_parts.push(parts[i].clone());

          i += 1;
        }
      }

      parts = new_parts;
    }

    parts
  }

  /// The set of all adjacent pairs of parts.
  fn parts_to_pairs(parts: &[String]) -> HashSet<(String, String)> {
    std::iter::zip(parts.iter().cloned(), parts.iter().skip(1).cloned()).collect()
  }

  /// Mapping of arbitrary bytes to printable chars.
  ///
  /// See: <https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/bpe.py#L22-L33>
  fn byte_to_char() -> HashMap<u8, char> {
    let mut printable: HashSet<u8> = (b'!'..=b'~').collect();
    printable.extend('¡' as u8..='¬' as u8);
    printable.extend('®' as u8..='ÿ' as u8);

    let mut map = HashMap::new();
    let mut n = 2u32.pow(8);

    for byte in u8::MIN..=u8::MAX {
      if !printable.contains(&byte) {
        map.insert(byte, char::from_u32(n).unwrap());

        n += 1;
      }
    }

    map.extend(printable.into_iter().map(|byte| (byte, byte as char)));

    map
  }
}
