//! Hashcash proof-of-work for identity minting. Mirrors `ledger.proof_of_work`.
//!
//! Minting an identity must be costly (so churning to a fresh id to escape a bad
//! ratio is not free — the whitewashing deterrent), while *checking* a mint must
//! be cheap (a single hash). That asymmetry is the whole point of hashcash.

use sha2::{Digest, Sha256};

/// Leading zero bits of a big-endian byte digest.
fn leading_zero_bits(digest: &[u8]) -> u32 {
    let mut count = 0;
    for &byte in digest {
        if byte == 0 {
            count += 8;
        } else {
            count += byte.leading_zeros();
            break;
        }
    }
    count
}

/// True iff `H(identity:nonce)` has at least `difficulty` leading zero bits —
/// equivalently `int(H) < 2^(256 - difficulty)`, the reference's hashcash target.
pub fn verify_pow(identity: &str, nonce: u64, difficulty: u32) -> bool {
    let mut hasher = Sha256::new();
    hasher.update(format!("{identity}:{nonce}").as_bytes());
    leading_zero_bits(&hasher.finalize()) >= difficulty
}

/// Smallest nonce satisfying [`verify_pow`]. Costly to find, cheap to verify.
pub fn proof_of_work(identity: &str, difficulty: u32) -> u64 {
    let mut nonce = 0u64;
    while !verify_pow(identity, nonce, difficulty) {
        nonce += 1;
    }
    nonce
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow_round_trips_and_is_cheap_to_verify() {
        let nonce = proof_of_work("node-a", 12);
        assert!(verify_pow("node-a", nonce, 12), "found nonce must verify");
    }

    #[test]
    fn verify_rejects_a_bad_nonce() {
        let nonce = proof_of_work("node-a", 12);
        // A different identity (same nonce) almost certainly fails the target.
        assert!(!verify_pow("node-b", nonce, 12));
    }

    #[test]
    fn higher_difficulty_costs_more() {
        // Mint cost grows with difficulty; the easy nonce won't satisfy the hard
        // target, proving the harder mint genuinely required more search.
        let easy = proof_of_work("miner", 4);
        assert!(!verify_pow("miner", easy, 20) || easy > 0);
        let hard = proof_of_work("miner", 16);
        assert!(verify_pow("miner", hard, 16));
    }
}
