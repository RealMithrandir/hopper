//! Simulated peer links with honest cost accounting. Mirrors
//! `reference/transport.py`.
//!
//! The only job here is to make the *physics* visible: how many bytes cross the
//! wire and how long that takes on a residential uplink. This is where layer
//! sharding visibly wins — the per-hop activation is constant and tiny while the
//! KV cache we deliberately *don't* ship grows without bound (Invariant 1).

/// A peer's link characteristics.
#[derive(Debug, Clone, Copy)]
pub struct LinkProfile {
    /// Round-trip latency in milliseconds (geo-clustered target < 20ms).
    pub rtt_ms: f64,
    /// Residential uplink in Mbps — the binding constraint.
    pub up_mbps: f64,
}

impl Default for LinkProfile {
    fn default() -> Self {
        Self {
            rtt_ms: 18.0,
            up_mbps: 30.0,
        }
    }
}

impl LinkProfile {
    /// Convenience constructor.
    pub fn new(rtt_ms: f64, up_mbps: f64) -> Self {
        Self { rtt_ms, up_mbps }
    }
}

/// Aggregated, human-readable transfer report (mirrors `NetworkMonitor.report`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkReport {
    pub hops: u64,
    pub activation_mb: f64,
    pub network_ms: f64,
    pub kv_ship_mb_avoided: f64,
}

fn round_to(x: f64, places: i32) -> f64 {
    let f = 10f64.powi(places);
    (x * f).round() / f
}

/// Accumulates bytes + simulated latency, plus the KV-bytes-avoided
/// counterfactual that makes the layer-sharding win measurable.
#[derive(Debug, Default, Clone)]
pub struct NetworkMonitor {
    pub total_bytes: u64,
    pub total_ms: f64,
    pub hops: u64,
    pub kv_ship_bytes_avoided: u64,
    /// Per-hop payload sizes, in order — instrumentation for the Invariant 1
    /// guard (every inter-stage hidden hop must be exactly `d_model * 4`).
    pub hop_bytes: Vec<usize>,
}

impl NetworkMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a transfer of `n_bytes` over `link`; returns the simulated one-way
    /// wall time in milliseconds.
    pub fn transfer(&mut self, n_bytes: usize, link: &LinkProfile) -> f64 {
        let serialize_ms = (n_bytes as f64 * 8.0) / (link.up_mbps * 1e6) * 1e3;
        let ms = link.rtt_ms / 2.0 + serialize_ms;
        self.total_bytes += n_bytes as u64;
        self.total_ms += ms;
        self.hops += 1;
        self.hop_bytes.push(n_bytes);
        ms
    }

    /// Record KV-cache bytes we did *not* have to ship for one stage.
    pub fn note_kv_avoided(&mut self, n_bytes: usize) {
        self.kv_ship_bytes_avoided += n_bytes as u64;
    }

    /// Aggregate report with the same rounding as the reference.
    pub fn report(&self) -> NetworkReport {
        NetworkReport {
            hops: self.hops,
            activation_mb: round_to(self.total_bytes as f64 / 1e6, 4),
            network_ms: round_to(self.total_ms, 1),
            kv_ship_mb_avoided: round_to(self.kv_ship_bytes_avoided as f64 / 1e6, 2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_accounts_bytes_and_latency() {
        let mut mon = NetworkMonitor::new();
        let link = LinkProfile::default();
        let ms = mon.transfer(512, &link);
        // rtt/2 + (512*8)/(30e6)*1e3
        assert!((ms - (9.0 + 4096.0 / 30e6 * 1e3)).abs() < 1e-9);
        assert_eq!(mon.total_bytes, 512);
        assert_eq!(mon.hops, 1);
        assert_eq!(mon.hop_bytes, vec![512]);
    }

    #[test]
    fn kv_avoided_counterfactual_accumulates() {
        let mut mon = NetworkMonitor::new();
        mon.note_kv_avoided(100_000);
        mon.note_kv_avoided(50_000);
        assert_eq!(mon.kv_ship_bytes_avoided, 150_000);
        assert!((mon.report().kv_ship_mb_avoided - 0.15).abs() < 1e-9);
    }
}
