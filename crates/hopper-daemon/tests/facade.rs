//! Phase 4a: the OpenAI-compatible facade, served by the in-process engine
//! (hopper-tiny). Verifies the response shape, usage accounting, the `hopper`
//! telemetry block, determinism (greedy), and an HTTP-level 200.

use std::path::PathBuf;

use hopper_daemon::facade::{
    build_app, generate_completion, spawn_engine, ChatMessage, ChatRequest,
};

fn golden_dir() -> String {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../reference/golden")
        .to_string_lossy()
        .into_owned()
}

fn user(content: &str, max_tokens: usize) -> ChatRequest {
    ChatRequest {
        model: "hopper-tiny".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: content.to_string(),
        }],
        max_tokens,
        temperature: 0.0,
    }
}

#[tokio::test]
async fn facade_returns_openai_shape_and_is_deterministic() {
    let handle = spawn_engine(&golden_dir(), 4).expect("spawn engine");

    let resp = generate_completion(&handle, user("explain the design", 12))
        .await
        .expect("completion");

    assert_eq!(resp.object, "chat.completion");
    assert_eq!(resp.choices.len(), 1);
    assert_eq!(resp.choices[0].index, 0);
    assert_eq!(resp.choices[0].message.role, "assistant");
    assert_eq!(resp.choices[0].finish_reason, "length");

    // Usage accounting is internally consistent.
    assert_eq!(resp.usage.completion_tokens, 12);
    assert_eq!(
        resp.usage.total_tokens,
        resp.usage.prompt_tokens + resp.usage.completion_tokens
    );

    // The non-standard hopper telemetry block is present; quiet verifier -> no bans.
    assert_eq!(resp.hopper.audit_fails, 0);

    // Greedy (temperature 0) is deterministic.
    let resp2 = generate_completion(&handle, user("explain the design", 12))
        .await
        .unwrap();
    assert_eq!(
        resp.choices[0].message.content,
        resp2.choices[0].message.content
    );
}

#[tokio::test]
async fn facade_http_route_returns_200_and_chat_completion() {
    use tower::ServiceExt; // oneshot

    let handle = spawn_engine(&golden_dir(), 4).expect("spawn engine");
    let app = build_app(handle);

    let body = serde_json::json!({
        "model": "hopper-tiny",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4
    })
    .to_string();

    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), 200);

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["usage"]["completion_tokens"], 4);
    assert!(v["hopper"].is_object(), "hopper telemetry block present");
}
