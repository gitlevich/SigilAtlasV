use crate::SidecarState;
use base64::Engine;
use serde::Serialize;
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

#[derive(Serialize)]
pub struct SigilEntry {
    pub name: String,
    pub folder_path: String,
    pub preview_data_url: Option<String>,
    pub modified_at: Option<f64>,
}

/// List `.sigil` directories one level under `workspace`. Each entry carries
/// the display name (from `collage.json` → `name`, falling back to the folder
/// stem) and the preview image as an inline data URL so the webview can render
/// it without a file-protocol or asset-scope round-trip.
#[tauri::command]
pub async fn list_sigils(workspace: String) -> Result<Vec<SigilEntry>, String> {
    let root = PathBuf::from(&workspace);
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let mut out: Vec<SigilEntry> = Vec::new();
    let entries = std::fs::read_dir(&root).map_err(|e| format!("read_dir failed: {e}"))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let ext_is_sigil = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.eq_ignore_ascii_case("sigil"))
            .unwrap_or(false);
        if !ext_is_sigil {
            continue;
        }

        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        let manifest_path = path.join("collage.json");
        let name = std::fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("name").and_then(|n| n.as_str()).map(String::from))
            .unwrap_or_else(|| stem.clone());

        let modified_at = std::fs::metadata(&manifest_path)
            .or_else(|_| std::fs::metadata(&path))
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64());

        let preview_path = path.join("preview.png");
        let preview_data_url = std::fs::read(&preview_path).ok().map(|bytes| {
            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
            format!("data:image/png;base64,{b64}")
        });

        out.push(SigilEntry {
            name,
            folder_path: path.to_string_lossy().to_string(),
            preview_data_url,
            modified_at,
        });
    }

    out.sort_by(|a, b| {
        b.modified_at
            .partial_cmp(&a.modified_at)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

/// Delete a `.sigil` directory. Refuses anything without the `.sigil`
/// extension to reduce blast radius of a mis-wired caller.
#[tauri::command]
pub async fn delete_sigil(folder_path: String) -> Result<(), String> {
    let path = PathBuf::from(&folder_path);
    let ext_is_sigil = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.eq_ignore_ascii_case("sigil"))
        .unwrap_or(false);
    if !ext_is_sigil {
        return Err(format!("refusing to delete non-.sigil path: {folder_path}"));
    }
    if !path.is_dir() {
        return Err(format!("not a directory: {folder_path}"));
    }
    std::fs::remove_dir_all(&path).map_err(|e| format!("delete failed: {e}"))
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
