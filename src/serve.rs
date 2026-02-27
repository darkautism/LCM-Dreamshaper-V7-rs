use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderValue, StatusCode, header::ACCEPT},
    response::{IntoResponse, Response},
    routing::{any, post},
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use rmcp::transport::{
    StreamableHttpServerConfig,
    streamable_http_server::{session::local::LocalSessionManager, tower::StreamableHttpService},
};
use serde::{Deserialize, Serialize};
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};

use crate::mcp::DreamshaperMcp;
use crate::pipeline::{GenerateRequest, Pipeline};

// ─────────────────────────────────────────────────────────────────────────────
// OpenAI-compatible request / response types
// https://platform.openai.com/docs/api-reference/images/create
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ImageGenRequest {
    pub prompt: String,
    /// Number of images (only 1 supported; others ignored)
    #[serde(default = "default_n")]
    pub n: u32,
    /// Only "512x512" is supported
    #[serde(default = "default_size")]
    pub size: String,
    /// "b64_json" (default) or "url" (not supported; falls back to b64_json)
    #[serde(default = "default_response_format")]
    pub response_format: String,
    /// Optional seed; omit or set null for a random seed
    pub seed: Option<u64>,
    /// Number of LCM inference steps (default 10)
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// Guidance scale (default 7.5)
    #[serde(default = "default_guidance_scale")]
    pub guidance_scale: f32,
}

fn default_n() -> u32 { 1 }
fn default_size() -> String { "512x512".to_string() }
fn default_response_format() -> String { "b64_json".to_string() }
fn default_steps() -> usize { 10 }
fn default_guidance_scale() -> f32 { 7.5 }

#[derive(Debug, Serialize)]
pub struct ImageData {
    pub b64_json: String,
    pub seed: u64,
}

#[derive(Debug, Serialize)]
pub struct ImageGenResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler
// ─────────────────────────────────────────────────────────────────────────────

async fn generate_images(
    State(pipeline): State<Arc<Mutex<Pipeline>>>,
    Json(req): Json<ImageGenRequest>,
) -> impl IntoResponse {
    let prompt = req.prompt.clone();
    let steps = req.steps;
    let guidance_scale = req.guidance_scale;
    let seed = req.seed;

    // Run heavy inference in a blocking thread so we don't block the async runtime
    let result = tokio::task::spawn_blocking(move || {
        let gen_req = GenerateRequest { prompt, steps, guidance_scale, seed };
        let locked = pipeline.lock().map_err(|e| anyhow::anyhow!("Lock poisoned: {}", e))?;
        locked.generate(gen_req)
    })
    .await;

    match result {
        Ok(Ok(result)) => {
            let b64 = BASE64.encode(&result.png_bytes);
            let created = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let resp = ImageGenResponse {
                created,
                data: vec![ImageData { b64_json: b64, seed: result.seed }],
            };
            (StatusCode::OK, Json(serde_json::to_value(resp).unwrap())).into_response()
        }
        Ok(Err(e)) => {
            eprintln!("❌ Generation error: {:#}", e);
            let resp = ErrorResponse {
                error: ErrorDetail {
                    message: e.to_string(),
                    r#type: "generation_error".to_string(),
                    code: None,
                },
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::to_value(resp).unwrap()))
                .into_response()
        }
        Err(e) => {
            eprintln!("❌ Task join error: {}", e);
            let resp = ErrorResponse {
                error: ErrorDetail {
                    message: "Internal task error".to_string(),
                    r#type: "internal_error".to_string(),
                    code: None,
                },
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::to_value(resp).unwrap()))
                .into_response()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MCP Accept-header shim
// Some MCP clients (e.g. VS Code Copilot) send only `Accept: application/json`
// without `text/event-stream`, causing rmcp to reject with 406 Not Acceptable.
// We use an axum handler that injects the header before calling StreamableHttpService.
// ─────────────────────────────────────────────────────────────────────────────

type McpService = StreamableHttpService<DreamshaperMcp, LocalSessionManager>;

#[derive(Clone)]
struct McpState {
    service: McpService,
    base_url: Arc<Mutex<String>>,
}

async fn mcp_handler(
    State(state): State<McpState>,
    mut req: axum::extract::Request,
) -> Response {
    // Update base_url from Host header so image URLs reflect the interface
    // the client actually connected through (LAN, Tailscale, localhost, etc.)
    if let Some(host) = req.headers().get(axum::http::header::HOST)
        .and_then(|v| v.to_str().ok())
    {
        if let Ok(mut base) = state.base_url.lock() {
            *base = format!("http://{}", host);
        }
    }

    // Inject Accept header so clients that omit text/event-stream still work
    req.headers_mut().insert(
        ACCEPT,
        HeaderValue::from_static("application/json, text/event-stream"),
    );
    let mut svc = state.service;
    match tower::Service::call(&mut svc, req).await {
        Ok(resp) => {
            let (parts, body) = resp.into_parts();
            axum::http::Response::from_parts(parts, axum::body::Body::new(body))
        }
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Server entry point
// ─────────────────────────────────────────────────────────────────────────────

pub async fn serve(host: &str, port: u16) -> Result<()> {
    let pipeline = tokio::task::block_in_place(Pipeline::load)?;
    let shared = Arc::new(Mutex::new(pipeline));

    // Ensure image output dir exists
    let image_dir = "/tmp/dreamshaper";
    std::fs::create_dir_all(image_dir)?;

    // base_url is dynamically updated from the Host header on each MCP request
    let base_url: Arc<Mutex<String>> = Arc::new(Mutex::new(
        format!("http://{}:{}", host, port)
    ));

    // MCP Streamable HTTP transport — one handler per session, sharing the pipeline.
    let mcp_shared = shared.clone();
    let mcp_base_url = base_url.clone();
    let mcp_service: McpService = StreamableHttpService::new(
            move || Ok(DreamshaperMcp::new(mcp_shared.clone(), mcp_base_url.clone())),
            Default::default(),
            StreamableHttpServerConfig {
                stateful_mode: true,
                sse_keep_alive: Some(std::time::Duration::from_secs(15)),
            },
        );

    let mcp_state = McpState { service: mcp_service, base_url };

    // Build a two-state router: pipeline for OpenAI REST, mcp_service for MCP.
    // The mcp_handler injects the Accept header shim so clients that omit
    // text/event-stream (e.g. VS Code Copilot) are handled transparently.
    let app = Router::new()
        .route("/v1/images/generations", post(generate_images))
        .with_state(shared)
        .merge(
            Router::new()
                .route("/mcp", any(mcp_handler))
                .route("/mcp/", any(mcp_handler))
                .with_state(mcp_state)
        )
        .nest_service("/images", ServeDir::new(image_dir))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    let addr = format!("{}:{}", host, port);
    eprintln!("🚀 Serving on http://{}/v1/images/generations  (OpenAI API)", addr);
    eprintln!("🤖 MCP endpoint:  http://{}/mcp  (Model Context Protocol)", addr);
    eprintln!("🖼️  Images:        http://{}/images/<seed>.png", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
