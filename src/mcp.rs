use std::sync::{Arc, Mutex};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use rmcp::{
    ServerHandler,
    ErrorData as McpError,
    handler::server::{
        router::tool::ToolRouter,
        tool::ToolCallContext,
        wrapper::Parameters,
    },
    model::{
        CallToolRequestParam, CallToolResult, ListToolsResult,
        PaginatedRequestParam, ServerCapabilities, ServerInfo,
    },
    service::{RequestContext, RoleServer},
    schemars, tool, tool_router,
};

use crate::pipeline::{GenerateRequest, Pipeline};

// ─────────────────────────────────────────────────────────────────────────────
// Tool input schema
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct GenerateImageParams {
    /// Text description of the image to generate
    pub prompt: String,
    /// Number of LCM denoising steps (1–50, default: 4 for fast preview)
    #[serde(default = "default_steps")]
    pub steps: usize,
    /// Classifier-free guidance scale (default: 7.5)
    #[serde(default = "default_guidance_scale")]
    pub guidance_scale: f32,
    /// Optional fixed seed for reproducibility; omit for random
    pub seed: Option<u64>,
}

fn default_steps() -> usize { 4 }
fn default_guidance_scale() -> f32 { 7.5 }

// ─────────────────────────────────────────────────────────────────────────────
// MCP handler
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct DreamshaperMcp {
    pipeline: Arc<Mutex<Pipeline>>,
    #[allow(dead_code)] // rmcp requires this field for tool routing internals
    tool_router: ToolRouter<Self>,
}

impl DreamshaperMcp {
    pub fn new(pipeline: Arc<Mutex<Pipeline>>) -> Self {
        Self {
            pipeline,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_router]
impl DreamshaperMcp {
    /// Generate an image from a text prompt using LCM Dreamshaper V7 on the RK3588 NPU.
    /// Returns a base64-encoded PNG string.
    #[tool(description = "Generate an image from a text prompt using LCM Dreamshaper V7 on Rockchip RK3588 NPU. Returns a base64-encoded PNG string and the seed used.")]
    fn generate_image(
        &self,
        Parameters(params): Parameters<GenerateImageParams>,
    ) -> Result<String, rmcp::ErrorData> {
        let pipeline = self.pipeline.clone();
        let req = GenerateRequest {
            prompt: params.prompt,
            steps: params.steps,
            guidance_scale: params.guidance_scale,
            seed: params.seed,
        };

        // Pipeline::generate is blocking (NPU inference); run directly since
        // rmcp calls tools via spawn_blocking internally.
        let result = pipeline
            .lock()
            .map_err(|e| rmcp::ErrorData::internal_error(format!("Lock poisoned: {e}"), None))?
            .generate(req)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        let b64 = BASE64.encode(&result.png_bytes);
        Ok(format!(
            "data:image/png;base64,{b64}\nseed={}",
            result.seed
        ))
    }
}

impl ServerHandler for DreamshaperMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "LCM Dreamshaper V7 image generation server running on Rockchip RK3588 NPU. \
                 Call generate_image with a text prompt to produce a 512×512 PNG image."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParam>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult {
            tools: self.tool_router.list_all(),
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        self.tool_router.call(ToolCallContext::new(self, request, context)).await
    }
}
