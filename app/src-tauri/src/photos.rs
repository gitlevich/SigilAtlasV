use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;

use tauri::{AppHandle, Manager, Runtime};

/// Locate the bundled sa-photos helper.
///
/// Release: copied into `Contents/MacOS/sa-photos-<triple>` by the Tauri bundler.
/// Dev: staged under `src-tauri/binaries/sa-photos-<triple>` by `build-photos.sh`.
/// The triple suffix is kept to stay consistent with Tauri's externalBin layout.
fn resolve_binary<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    // Tauri's externalBin stages the binary with a rust-triple suffix.
    // std::env::consts::ARCH maps cleanly to the Apple triple we need.
    let arch = std::env::consts::ARCH;
    let name = format!("sa-photos-{arch}-apple-darwin");

    // Release path — inside the .app bundle.
    if let Ok(exe_dir) = std::env::current_exe().and_then(|p| {
        p.parent()
            .map(|d| d.to_path_buf())
            .ok_or(std::io::Error::other("no parent"))
    }) {
        let candidate = exe_dir.join(&name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    // Dev path — staged binary under src-tauri/binaries/.
    let resource = app
        .path()
        .resolve(format!("binaries/{name}"), tauri::path::BaseDirectory::Resource);
    if let Ok(p) = resource {
        if p.is_file() {
            return Ok(p);
        }
    }

    // Fallback: CARGO_MANIFEST_DIR for `cargo tauri dev` before resources are staged.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = manifest.join("binaries").join(&name);
    if candidate.is_file() {
        return Ok(candidate);
    }

    Err(format!("sa-photos helper not found (looked for {name})"))
}

/// Result of the auth subcommand: the single lowercase status token printed
/// on stdout (authorized, limited, denied, restricted, notdetermined).
#[derive(serde::Serialize)]
pub struct AuthResult {
    pub status: String,
}

/// Request or check Photos library authorization. Blocks on the native prompt
/// the first time. Subsequent calls return the cached grant.
#[tauri::command]
pub async fn photos_auth<R: Runtime>(app: AppHandle<R>) -> Result<AuthResult, String> {
    let bin = resolve_binary(&app)?;
    let output = Command::new(&bin)
        .arg("auth")
        .output()
        .await
        .map_err(|e| format!("spawn sa-photos failed: {e}"))?;
    let status = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(AuthResult { status })
}

/// Summary of a completed photos enumerate + thumbnail pass.
#[derive(serde::Serialize, Default)]
pub struct EnumerateSummary {
    pub registered: usize,
    pub skipped: usize,
    pub thumbnails: usize,
    pub thumbnail_failures: usize,
}

/// Ingest reply from Python: the (image_id, local_identifier) pairs that
/// need a thumbnail, the absolute directory to write them into, and the
/// cancelled flag that signals the enumerate loop to unwind.
#[derive(serde::Deserialize)]
struct IngestReply {
    assigned: Vec<(String, String)>,
    skipped: usize,
    thumbnails_dir: String,
    #[serde(default)]
    cancelled: bool,
}

/// Enumerate the System photo library and drive thumbnail generation.
/// Records stream from the helper's stdout to Python in 500-photo batches;
/// each batch is registered and then thumbnails for it are produced by one
/// `sa-photos thumbs` process that consumes the batch on stdin in parallel.
#[tauri::command]
pub async fn photos_enumerate<R: Runtime>(
    app: AppHandle<R>,
    state: tauri::State<'_, crate::SidecarState>,
    since: Option<f64>,
) -> Result<EnumerateSummary, String> {
    let bin = resolve_binary(&app)?;
    let port = state
        .port
        .lock()
        .await
        .ok_or("Python sidecar not started")?;

    let client = reqwest::Client::new();
    let base = format!("http://127.0.0.1:{port}");

    // Session/start before any sa-photos spawn so the UI shows "running"
    // immediately and so we fail fast if another import is in flight.
    let resp = client
        .post(format!("{base}/sources/photos/session/start"))
        .json(&serde_json::json!({}))
        .send()
        .await
        .map_err(|e| format!("photos session/start failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "photos session/start {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        ));
    }

    // Run the enumerate loop. Errors here must still close the session so
    // the UI transitions out of "running"; defer the close through a local
    // result binding rather than an early return.
    let mut summary = EnumerateSummary::default();
    let result = enumerate_inner(&bin, &client, &base, since, &mut summary).await;

    let complete_body = serde_json::json!({
        "registered": summary.registered,
        "skipped": summary.skipped,
        "thumbnails": summary.thumbnails,
        "thumbnail_failures": summary.thumbnail_failures,
        "error": result.as_ref().err().cloned(),
    });
    if let Err(e) = client
        .post(format!("{base}/sources/photos/session/complete"))
        .json(&complete_body)
        .send()
        .await
    {
        eprintln!("[photos] session/complete failed: {e}");
    }

    result.map(|_| summary)
}

async fn enumerate_inner(
    bin: &Path,
    client: &reqwest::Client,
    base: &str,
    since: Option<f64>,
    summary: &mut EnumerateSummary,
) -> Result<(), String> {
    let ingest_url = format!("{base}/sources/photos/ingest");
    let thumbs_url = format!("{base}/sources/photos/thumbnails-generated");

    let mut cmd = Command::new(bin);
    cmd.arg("enumerate");
    if let Some(s) = since {
        cmd.arg("--since").arg(s.to_string());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::inherit());

    let mut child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let mut reader = BufReader::new(stdout).lines();

    let total_url = format!("{base}/sources/photos/session/total");
    let mut batch: Vec<serde_json::Value> = Vec::with_capacity(500);
    while let Some(line) = reader
        .next_line()
        .await
        .map_err(|e| format!("read sa-photos stdout: {e}"))?
    {
        let record: serde_json::Value = serde_json::from_str(&line)
            .map_err(|e| format!("bad NDJSON from sa-photos: {e}; line={line}"))?;
        // Count preamble: first line from enumerate carries {"type":"count","total":N}
        // so progress bars can size before records flow.
        if record.get("type").and_then(|v| v.as_str()) == Some("count") {
            let total = record.get("total").and_then(|v| v.as_u64()).unwrap_or(0);
            let _ = client
                .post(&total_url)
                .json(&serde_json::json!({ "total": total }))
                .send()
                .await;
            continue;
        }
        batch.push(record);
        if batch.len() >= 500 {
            let cancelled = register_and_thumbnail(
                client, &ingest_url, &thumbs_url, bin,
                std::mem::take(&mut batch), summary,
            ).await?;
            if cancelled {
                // Kill the enumerate child so PHPhotoLibrary iteration stops
                // and the process doesn't keep pushing records into our pipe.
                let _ = child.kill().await;
                return Err("cancelled".to_string());
            }
        }
    }
    if !batch.is_empty() {
        let cancelled = register_and_thumbnail(
            client, &ingest_url, &thumbs_url, bin, batch, summary,
        ).await?;
        if cancelled {
            let _ = child.kill().await;
            return Err("cancelled".to_string());
        }
    }

    let status = child
        .wait()
        .await
        .map_err(|e| format!("wait sa-photos: {e}"))?;
    if !status.success() {
        return Err(format!("sa-photos enumerate exited {status}"));
    }
    Ok(())
}

async fn register_and_thumbnail(
    client: &reqwest::Client,
    ingest_url: &str,
    thumbs_url: &str,
    bin: &std::path::Path,
    batch: Vec<serde_json::Value>,
    summary: &mut EnumerateSummary,
) -> Result<bool, String> {
    let body = serde_json::json!({ "records": batch });
    let resp = client
        .post(ingest_url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("ingest POST failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!(
            "ingest POST {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        ));
    }
    let reply: IngestReply = resp
        .json()
        .await
        .map_err(|e| format!("ingest reply parse failed: {e}"))?;
    if reply.cancelled {
        return Ok(true);
    }
    summary.registered += reply.assigned.len();
    summary.skipped += reply.skipped;

    let thumbs_dir = std::path::PathBuf::from(&reply.thumbnails_dir);
    std::fs::create_dir_all(&thumbs_dir)
        .map_err(|e| format!("mkdir thumbnails_dir: {e}"))?;

    let succeeded = run_thumbs_batch(bin, &thumbs_dir, &reply.assigned, summary).await?;
    summary.thumbnails += succeeded.len();

    if !succeeded.is_empty() {
        let body = serde_json::json!({ "image_ids": succeeded });
        let resp = client
            .post(thumbs_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("thumbnails-generated POST failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!(
                "thumbnails-generated POST {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            ));
        }
    }
    Ok(false)
}

/// Result of one thumbnail write, parsed from `sa-photos thumbs` stdout.
#[derive(serde::Deserialize)]
struct ThumbResult {
    out: String,
    ok: bool,
    #[serde(default)]
    err: Option<String>,
}

const THUMB_SIZE: u32 = 512;

/// Generate thumbnails for a batch of assigned photos by piping NDJSON work
/// items into a single long-lived `sa-photos thumbs` process. Inside that
/// process PhotoKit requests run in parallel via `concurrentPerform`, so the
/// whole batch completes in one framework-load's worth of startup cost.
async fn run_thumbs_batch(
    bin: &Path,
    thumbs_dir: &Path,
    assigned: &[(String, String)],
    summary: &mut EnumerateSummary,
) -> Result<Vec<String>, String> {
    if assigned.is_empty() {
        return Ok(Vec::new());
    }

    // out-path → image_id so we can reconcile the result stream back to ids.
    let mut by_out: HashMap<String, String> = HashMap::with_capacity(assigned.len());
    let mut stdin_buf = String::with_capacity(assigned.len() * 160);
    for (image_id, local_id) in assigned {
        let out = thumbs_dir.join(format!("{image_id}.jpg"));
        let out_str = out.to_string_lossy().to_string();
        let item = serde_json::json!({
            "id": local_id,
            "size": THUMB_SIZE,
            "out": out_str,
        });
        stdin_buf.push_str(&item.to_string());
        stdin_buf.push('\n');
        by_out.insert(out_str, image_id.clone());
    }

    let mut child = Command::new(bin)
        .arg("thumbs")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("spawn sa-photos thumbs: {e}"))?;

    let mut stdin = child.stdin.take().ok_or("no stdin on sa-photos thumbs")?;
    let writer = tokio::spawn(async move {
        let _ = stdin.write_all(stdin_buf.as_bytes()).await;
        let _ = stdin.shutdown().await;
    });

    let stdout = child.stdout.take().ok_or("no stdout on sa-photos thumbs")?;
    let mut reader = BufReader::new(stdout).lines();

    let mut succeeded: Vec<String> = Vec::with_capacity(assigned.len());
    while let Some(line) = reader
        .next_line()
        .await
        .map_err(|e| format!("read thumbs stdout: {e}"))?
    {
        let result: ThumbResult = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("[photos] bad thumbs line: {e}; line={line}");
                continue;
            }
        };
        if result.ok {
            if let Some(id) = by_out.remove(&result.out) {
                succeeded.push(id);
            }
        } else {
            summary.thumbnail_failures += 1;
            if let Some(err) = &result.err {
                eprintln!("[photos] thumb failed {}: {err}", result.out);
            }
        }
    }

    let _ = writer.await;
    let status = child
        .wait()
        .await
        .map_err(|e| format!("wait thumbs: {e}"))?;
    if !status.success() {
        return Err(format!("sa-photos thumbs exited {status}"));
    }
    Ok(succeeded)
}

/// Write a single JPEG thumbnail for a Photos asset to disk. Called lazily
/// by the renderer when a tile backed by a `photos://…` source_path becomes
/// visible and its thumbnail file does not yet exist.
#[tauri::command]
pub async fn photos_thumb<R: Runtime>(
    app: AppHandle<R>,
    local_identifier: String,
    size: u32,
    out_path: String,
) -> Result<(), String> {
    let bin = resolve_binary(&app)?;
    let status = Command::new(&bin)
        .args(["thumb", &local_identifier, &size.to_string(), &out_path])
        .status()
        .await
        .map_err(|e| format!("spawn sa-photos thumb: {e}"))?;
    if !status.success() {
        return Err(format!("sa-photos thumb exited {status}"));
    }
    Ok(())
}
