use std::collections::{HashMap, HashSet};

use fancy_regex::Regex;

pub type Token = u64;

pub struct Encoder {
  /// Matches "words" on whitespace, contracition, or numeric boundaries.
  word_re: Regex,
  /// Maps arbitrary bytes to printable "latent" characters.
  byte_to_char: HashMap<u8, char>,
  /// Maps "latent" characters back to their original bytes.
  char_to_byte: HashMap<char, u8>,

  bpe_ranks: HashMap<(String, String), usize>,

  /// BPE split cache.
  cache: HashMap<String, Vec<String>>,
}

impl Encoder {
  pub fn new() -> Self {
    let byte_to_char = Self::byte_to_char();

    Self {
      word_re: Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+").unwrap(),
      char_to_byte: byte_to_char.iter().map(|(b, c)| (*c, *b)).collect(),
      bpe_ranks: HashMap::new(),
      byte_to_char,
      cache: HashMap::new(),
    }
  }

  /// Tokenizes a string into into it's BPE strings. Splits on all characters
  /// then performs all BPE merges.
  ///
  /// For example, "elephant" -> ["ele", "phant"]
  fn tokenize(&mut self, word: String) -> Vec<String> {
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

  pub fn encode(&self, text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    for word in self.word_re.find_iter(text) {
      let word: String = word
        .unwrap()
        .as_str()
        .as_bytes()
        .iter()
        .map(|b| self.byte_to_char.get(b).unwrap())
        .collect();

      // tokens.extend(self.bpe(word).into_iter().map(|bpe_token| self.bpe_encoder.get(bpe_token).unwrap()))
    }

    tokens
  }

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

#[cfg(test)]
mod tests {
  #[test]
  fn exploration() {
    assert_eq!(2 + 2, 4);
  }

  #[test]
  fn another() {
    panic!("Make this test fail");
  }
}
