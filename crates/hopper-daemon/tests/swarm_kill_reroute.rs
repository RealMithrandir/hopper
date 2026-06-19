//! Phase 3 gate: 2+ OS processes form a libp2p swarm, complete an inference
//! across them, and survive a worker killed mid-stream (coordinator reroutes to
//! the backup and finishes the generation).
//!
//! The daemon binary path is injected by Cargo as `CARGO_BIN_EXE_hopper-daemon`.

use std::io::BufRead;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const BIN: &str = env!("CARGO_BIN_EXE_hopper-daemon");
const N_STAGES: &str = "4";
const MAX_TOKENS: usize = 20;

fn golden_dir() -> String {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../reference/golden")
        .to_string_lossy()
        .into_owned()
}

/// Children killed (and reaped) on drop, so a panicking assertion never leaks.
struct Procs(Vec<Child>);
impl Drop for Procs {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

type Lines = Arc<Mutex<Vec<String>>>;

/// Drain a child's stdout into a shared line buffer on a background thread.
fn reader(child: &mut Child) -> Lines {
    let stdout = child.stdout.take().expect("piped stdout");
    let lines: Lines = Arc::new(Mutex::new(Vec::new()));
    let sink = lines.clone();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout)
            .lines()
            .map_while(Result::ok)
        {
            sink.lock().unwrap().push(line);
        }
    });
    lines
}

/// Poll `lines` until `pred` holds or `timeout` elapses.
fn wait_for(lines: &Lines, pred: impl Fn(&[String]) -> bool, timeout: Duration) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if pred(&lines.lock().unwrap()) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    false
}

fn line_starting(lines: &Lines, prefix: &str) -> Option<String> {
    lines
        .lock()
        .unwrap()
        .iter()
        .find(|l| l.starts_with(prefix))
        .cloned()
}

fn spawn_worker(seed: &str) -> Child {
    Command::new(BIN)
        .args([
            "worker",
            "--golden",
            &golden_dir(),
            "--stages",
            "0,1,2,3",
            "--n-stages",
            N_STAGES,
            "--seed",
            seed,
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn worker")
}

/// Parse a worker's `PEER <id> ADDR <multiaddr>` handshake line.
fn peer_addr(lines: &Lines) -> (String, String) {
    let line = line_starting(lines, "PEER ").expect("worker PEER line");
    let parts: Vec<&str> = line.split_whitespace().collect();
    (parts[1].to_string(), parts[3].to_string())
}

#[test]
fn swarm_completes_inference_and_survives_midstream_kill() {
    // Two full-replica workers (each hosts all 4 stages) so any stage can reroute.
    let mut w1 = spawn_worker("1");
    let w1_lines = reader(&mut w1);
    let mut w2 = spawn_worker("2");
    let w2_lines = reader(&mut w2);
    let mut procs = Procs(vec![w1, w2]); // index 0 = w1, 1 = w2

    assert!(
        wait_for(
            &w1_lines,
            |l| l.iter().any(|x| x.starts_with("PEER")),
            Duration::from_secs(25)
        ),
        "worker 1 never announced"
    );
    assert!(
        wait_for(
            &w2_lines,
            |l| l.iter().any(|x| x.starts_with("PEER")),
            Duration::from_secs(25)
        ),
        "worker 2 never announced"
    );
    let (p1, a1) = peer_addr(&w1_lines);
    let (p2, a2) = peer_addr(&w2_lines);

    // Coordinator drives the inference; a small per-token delay gives us a window
    // to land the kill mid-stream.
    let mut coord = Command::new(BIN)
        .args([
            "coordinator",
            "--bootstrap",
            &format!("{p1}@{a1}"),
            "--bootstrap",
            &format!("{p2}@{a2}"),
            "--n-stages",
            N_STAGES,
            "--prompt",
            "kill test",
            "--max-tokens",
            &MAX_TOKENS.to_string(),
            "--step-delay-ms",
            "150",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn coordinator");
    let c_lines = reader(&mut coord);
    procs.0.push(coord);

    // Discovery succeeded and generation is under way.
    assert!(
        wait_for(
            &c_lines,
            |l| l.iter().any(|x| x == "TOKEN 1"),
            Duration::from_secs(40)
        ),
        "inference did not start across the swarm"
    );

    // Kill the worker the coordinator is actively using.
    let primary = line_starting(&c_lines, "PRIMARY ")
        .and_then(|l| l.split_whitespace().nth(1).map(str::to_string))
        .expect("coordinator PRIMARY line");
    let victim = if primary == p1 { 0 } else { 1 };
    procs.0[victim].kill().expect("kill primary worker");

    // The coordinator must reroute and still finish all tokens.
    assert!(
        wait_for(
            &c_lines,
            |l| l.iter().any(|x| x.starts_with("DONE")),
            Duration::from_secs(40)
        ),
        "coordinator did not finish after the mid-stream kill"
    );

    let done = line_starting(&c_lines, "DONE").unwrap();
    assert!(
        done.contains(&format!("tokens={MAX_TOKENS}")),
        "did not complete all tokens: {done}"
    );

    let reroutes: usize = done
        .split_whitespace()
        .find_map(|t| t.strip_prefix("reroutes="))
        .and_then(|n| n.parse().ok())
        .expect("reroutes field");
    assert!(reroutes >= 1, "expected a reroute after the kill: {done}");
    assert!(
        line_starting(&c_lines, "REROUTE").is_some(),
        "expected an explicit REROUTE event"
    );
}
