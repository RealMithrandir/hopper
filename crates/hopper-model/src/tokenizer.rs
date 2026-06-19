//! Byte-level tokenizer (mirrors `model.encode` / `model.decode`). Zero
//! downloads: a token *is* a UTF-8 byte, so `vocab_size == 256`.

/// Encode text into byte-token ids.
pub fn encode(text: &str) -> Vec<usize> {
    text.as_bytes().iter().map(|&b| b as usize).collect()
}

/// Decode byte-token ids back into text (lossy on invalid UTF-8, matching the
/// reference's `errors="replace"`).
pub fn decode(ids: &[usize]) -> String {
    let bytes: Vec<u8> = ids.iter().map(|&i| (i % 256) as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_round_trips_ascii() {
        assert_eq!(encode("hopper"), vec![104, 111, 112, 112, 101, 114]);
        assert_eq!(decode(&encode("hello")), "hello");
    }
}
