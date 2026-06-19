//! A pluggable clock, mirroring the reference ledger's injectable `clock`
//! callable. Production uses [`SystemClock`]; tests use [`ManualClock`] to make
//! time-decay deterministic and instantaneous.

use std::cell::Cell;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Source of "now" in fractional seconds since the Unix epoch.
pub trait Clock {
    fn now(&self) -> f64;
}

/// Wall-clock time.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    }
}

/// A settable test clock. Cloned handles share the same underlying time, so a
/// test can advance time after handing a clone to the ledger.
#[derive(Clone, Default)]
pub struct ManualClock(Rc<Cell<f64>>);

impl ManualClock {
    /// A clock fixed at `t` seconds.
    pub fn new(t: f64) -> Self {
        Self(Rc::new(Cell::new(t)))
    }

    /// Jump to absolute time `t`.
    pub fn set(&self, t: f64) {
        self.0.set(t);
    }

    /// Move forward by `dt` seconds.
    pub fn advance(&self, dt: f64) {
        self.0.set(self.0.get() + dt);
    }
}

impl Clock for ManualClock {
    fn now(&self) -> f64 {
        self.0.get()
    }
}
