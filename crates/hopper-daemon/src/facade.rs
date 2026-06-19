//! Layer 3: an OpenAI-compatible facade over the swarm (mirrors `reference/api.py`).
//!
//! Phase 4a serves it from an **in-process engine** running `hopper-tiny` (the
//! across-processes real-model path is Phase 4b). The engine is not `Send`, so it
//! lives on a dedicated thread built from leaked-`'static` weights and is reached
//! over a channel ([`GenHandle`]) — no shared-lock state on the async side.

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use hopper_engine::{Engine, Node, NodeMap};
use hopper_ledger::Ledger;
use hopper_model::{encode, golden::Golden, shard, ModelConfig, Stage, Weights};
use hopper_net::{LinkProfile, Router as StageRouter};
use hopper_verify::Verifier;

use crate::ServeArgs;

// ----- engine worker thread -------------------------------------------------

/// Result of one generation, flattened for the HTTP layer.
pub struct GenOutput {
    pub text: String,
    pub completion_tokens: usize,
    pub prompt_tokens: usize,
    pub audits: usize,
    pub audit_fails: usize,
    pub reroutes: usize,
}

struct GenRequest {
    prompt: String,
    max_tokens: usize,
    temperature: f64,
    resp: oneshot::Sender<Result<GenOutput, String>>,
}

/// A cloneable handle to the in-process generation engine.
#[derive(Clone)]
pub struct GenHandle {
    tx: mpsc::UnboundedSender<GenRequest>,
}

impl GenHandle {
    /// Generate from a prompt; serialized on the engine thread.
    pub async fn generate(
        &self,
        prompt: String,
        max_tokens: usize,
        temperature: f64,
    ) -> Result<GenOutput, String> {
        let (resp, rx) = oneshot::channel();
        self.tx
            .send(GenRequest {
                prompt,
                max_tokens,
                temperature,
                resp,
            })
            .map_err(|_| "engine thread gone".to_string())?;
        rx.await
            .map_err(|_| "engine dropped the request".to_string())?
    }
}

/// Stand up a single-process swarm (all stages local) reachable via [`GenHandle`].
/// Weights are loaded here (they are `Send`) and leaked so the engine — which is
/// not `Send` — can be built and owned entirely on its own thread.
pub fn spawn_engine(golden_dir: &str, n_stages: usize) -> Result<GenHandle> {
    let golden = Golden::load(golden_dir)?;
    let cfg = golden.config();
    let weights: &'static Weights = Box::leak(Box::new(golden.weights()?));
    let (tx, mut rx) = mpsc::unbounded_channel::<GenRequest>();

    thread::Builder::new()
        .name("hopper-engine".into())
        .spawn(move || {
            let mut engine = build_inprocess_engine(&cfg, weights, n_stages);
            while let Some(req) = rx.blocking_recv() {
                let result = engine
                    .generate(&req.prompt, req.max_tokens, "client", req.temperature)
                    .map(|(text, ids, stats)| GenOutput {
                        text,
                        completion_tokens: ids.len(),
                        prompt_tokens: encode(&req.prompt).len(),
                        audits: stats.audits,
                        audit_fails: stats.audit_fails,
                        reroutes: stats.reroutes,
                    })
                    .map_err(|e| e.to_string());
                let _ = req.resp.send(result);
            }
        })?;

    Ok(GenHandle { tx })
}

fn build_inprocess_engine(
    cfg: &ModelConfig,
    weights: &'static Weights,
    n_stages: usize,
) -> Engine<'static> {
    let bounds = shard(cfg, n_stages);
    let mut ledger = Ledger::default();
    let mut router = StageRouter::new();
    let mut nodes = NodeMap::new();

    for (sid, &(lo, hi)) in bounds.iter().enumerate() {
        let id = format!("local-{sid}");
        ledger.register(&id, 4);
        let mut node = Node::new(&id, LinkProfile::default(), sid as u64);
        node.host(sid, Stage::new(cfg.clone(), weights, lo, hi));
        router.announce(&id, sid, 10.0);
        nodes.insert(id, node);
    }
    ledger.register("client", 4);

    // Quiet verifier: the facade just serves; auditing rides the swarm path.
    let verifier = Verifier::new(cfg.clone(), weights, n_stages, 0.0, 2e-3, 0.0, 0);
    Engine::new(cfg.clone(), nodes, router, ledger, verifier, n_stages, 7)
}

// ----- OpenAI-compatible HTTP shapes ---------------------------------------

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

fn default_model() -> String {
    "hopper-tiny".to_string()
}
fn default_max_tokens() -> usize {
    64
}

#[derive(Deserialize)]
pub struct ChatRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: f64,
}

#[derive(Serialize)]
pub struct AssistantMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Non-standard `hopper` telemetry block (swarm extension).
#[derive(Serialize)]
pub struct HopperTelemetry {
    pub audits: usize,
    pub audit_fails: usize,
    pub reroutes: usize,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    pub hopper: HopperTelemetry,
}

static COMPLETION_COUNTER: AtomicU64 = AtomicU64::new(0);

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Core handler logic (no HTTP layer), so it can be unit-tested directly.
pub async fn generate_completion(
    handle: &GenHandle,
    req: ChatRequest,
) -> Result<ChatResponse, String> {
    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let out = handle
        .generate(prompt, req.max_tokens, req.temperature)
        .await?;

    let id = COMPLETION_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(ChatResponse {
        id: format!("chatcmpl-{id:012x}"),
        object: "chat.completion".to_string(),
        created: now_secs(),
        model: req.model,
        choices: vec![Choice {
            index: 0,
            message: AssistantMessage {
                role: "assistant".to_string(),
                content: out.text,
            },
            finish_reason: "length".to_string(),
        }],
        usage: Usage {
            prompt_tokens: out.prompt_tokens,
            completion_tokens: out.completion_tokens,
            total_tokens: out.prompt_tokens + out.completion_tokens,
        },
        hopper: HopperTelemetry {
            audits: out.audits,
            audit_fails: out.audit_fails,
            reroutes: out.reroutes,
        },
    })
}

async fn chat_handler(State(handle): State<GenHandle>, Json(req): Json<ChatRequest>) -> Response {
    match generate_completion(&handle, req).await {
        Ok(resp) => (StatusCode::OK, Json(resp)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e })),
        )
            .into_response(),
    }
}

/// Build the axum router exposing `POST /v1/chat/completions`.
pub fn build_app(handle: GenHandle) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_handler))
        .with_state(handle)
}

/// Run the `serve` role: in-process engine + HTTP facade.
pub async fn run(args: ServeArgs) -> Result<()> {
    let handle = spawn_engine(&args.golden, args.n_stages)?;
    let app = build_app(handle);
    let listener = tokio::net::TcpListener::bind(&args.bind).await?;
    let addr = listener.local_addr()?;
    println!("LISTENING {addr}");
    tracing::info!(%addr, "OpenAI facade serving");
    axum::serve(listener, app).await?;
    Ok(())
}
