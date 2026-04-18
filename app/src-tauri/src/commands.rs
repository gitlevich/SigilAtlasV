use crate::SidecarState;
use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};
use tauri::State;

/// Buffer of `.sigil` paths the OS asked us to open before the webview was
/// ready. Drained via the `drain_pending_opens` command on JS startup.
pub struct PendingOpens(pub Arc<StdMutex<Vec<String>>>);

#[tauri::command]
pub fn drain_pending_opens(state: State<'_, PendingOpens>) -> Vec<String> {
    state
        .0
        .lock()
        .ok()
        .map(|mut v| std::mem::take(&mut *v))
        .unwrap_or_default()
}

/// Resolve a path relative to the project root (two levels up from src-tauri).
fn project_root() -> PathBuf {
    let exe = std::env::current_exe().expect("cannot find exe path");
    // In dev: target/debug/sigil-atlas -> go up to src-tauri, then app, then project root
    // In release: similar structure
    let mut dir = exe.parent().unwrap().to_path_buf();

    // Walk up until we find "workspace" dir or hit root
    for _ in 0..10 {
        if dir.join("workspace").is_dir() {
            return dir;
        }
        if let Some(parent) = dir.parent() {
            dir = parent.to_path_buf();
        } else {
            break;
        }
    }

    // Fallback: assume CWD is project root
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[tauri::command]
pub async fn start_sidecar(
    state: State<'_, SidecarState>,
    workspace: String,
    python: String,
) -> Result<u16, String> {
    let root = project_root();
    let ws_path = root.join(&workspace);
    let py_path = root.join(&python);

    let ws = if ws_path.is_dir() {
        ws_path.to_string_lossy().to_string()
    } else {
        workspace
    };

    let py = if py_path.is_file() {
        py_path.to_string_lossy().to_string()
    } else {
        python
    };

    let (child, port) = crate::sidecar::spawn_sidecar(&ws, &py).await?;
    *state.port.lock().await = Some(port);
    *state.process.lock().await = Some(child);
    *state.workspace.lock().await = Some(ws);
    Ok(port)
}

#[tauri::command]
pub async fn get_sidecar_port(state: State<'_, SidecarState>) -> Result<u16, String> {
    state
        .port
        .lock()
        .await
        .ok_or_else(|| "Sidecar not started".to_string())
}

#[tauri::command]
pub async fn proxy_get(
    state: State<'_, SidecarState>,
    path: String,
) -> Result<serde_json::Value, String> {
    let port = state
        .port
        .lock()
        .await
        .ok_or("Sidecar not started")?;

    let url = format!("http://127.0.0.1:{port}{path}");
    let resp = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {e}"))?;
    resp.json()
        .await
        .map_err(|e| format!("JSON parse failed: {e}"))
}

#[tauri::command]
pub async fn proxy_post(
    state: State<'_, SidecarState>,
    path: String,
    body: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let port = state
        .port
        .lock()
        .await
        .ok_or("Sidecar not started")?;

    let url = format!("http://127.0.0.1:{port}{path}");
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;
    resp.json()
        .await
        .map_err(|e| format!("JSON parse failed: {e}"))
}
