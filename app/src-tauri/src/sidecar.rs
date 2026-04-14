use std::process::Stdio;
use tokio::io::AsyncBufReadExt;
use tokio::process::Command;

/// Spawn the Python sidecar and return (child, port).
pub async fn spawn_sidecar(workspace: &str, python: &str) -> Result<(tokio::process::Child, u16), String> {
    let mut child = Command::new(python)
        .args(["-m", "sigil_atlas.serve", "--workspace", workspace])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to spawn sidecar: {e}"))?;

    // Read first line from stdout to get the port
    let stdout = child.stdout.take().ok_or("No stdout from sidecar")?;
    let mut reader = tokio::io::BufReader::new(stdout);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .await
        .map_err(|e| format!("Failed to read sidecar port: {e}"))?;

    let port: u16 = line
        .trim()
        .parse()
        .map_err(|e| format!("Invalid port from sidecar: {line:?} ({e})"))?;

    Ok((child, port))
}
