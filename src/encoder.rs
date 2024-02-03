use std::collections::{HashMap, HashSet};

pub fn bytes_to_unicode() -> HashMap<char, u8> {
  let mut printable: HashSet<u8> = (b'!'..=b'~').collect();
  printable.extend('¡' as u8..='¬' as u8);
  printable.extend('®' as u8..='ÿ' as u8);

  let mut map = HashMap::new();
  let mut n = 2u32.pow(8);

  for byte in 0..=u8::MAX {
    if !printable.contains(&byte) {
      map.insert(char::from_u32(n).unwrap(), byte);

      n += 1;
    }
  }

  map.extend(printable.into_iter().map(|byte| (byte as char, byte)));

  map
}
