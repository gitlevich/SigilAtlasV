use std::path::Path;
use std::process::Stdio;
use tokio::io::AsyncBufReadExt;
use tokio::process::Command;

const PID_FILE: &str = "sidecar.pid";

/// Kill any stale sidecar left from a previous run.
///
/// Reads the PID file from the workspace directory, verifies the process
/// is actually a sigil_atlas sidecar (not some unrelated reuse of the PID),
/// and kills it. Removes the PID file afterward regardless.
pub fn kill_stale(workspace: &str) {
    let pid_path = Path::new(workspace).join(PID_FILE);
    let Ok(contents) = std::fs::read_to_string(&pid_path) else {
        return;
    };
    let _ = std::fs::remove_file(&pid_path);

    let Ok(pid) = contents.trim().parse::<u32>() else {
        return;
    };

    // Verify this PID is actually our sidecar before killing it.
    // Read /proc/<pid>/cmdline on Linux, or use `ps -p` on macOS.
    if !is_sigil_sidecar(pid) {
        return;
    }

    eprintln!("[sidecar] killing stale sidecar (pid {pid})");
    unsafe {
        libc::kill(pid as i32, libc::SIGTERM);
    }

    // Give it a moment to exit, then force-kill if needed
    std::thread::sleep(std::time::Duration::from_millis(500));
    unsafe {
        // kill(pid, 0) checks if process still exists
        if libc::kill(pid as i32, 0) == 0 {
            eprintln!("[sidecar] force-killing stale sidecar (pid {pid})");
            libc::kill(pid as i32, libc::SIGKILL);
        }
    }
}

/// Check if a PID belongs to a sigil_atlas.serve process.
fn is_sigil_sidecar(pid: u32) -> bool {
    let output = std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output();

    match output {
        Ok(out) => {
            let cmdline = String::from_utf8_lossy(&out.stdout);
            cmdline.contains("sigil_atlas.serve")
        }
        Err(_) => false,
    }
}

/// Write the sidecar PID to the workspace directory.
fn write_pid(workspace: &str, pid: u32) {
    let pid_path = Path::new(workspace).join(PID_FILE);
    if let Err(e) = std::fs::write(&pid_path, pid.to_string()) {
        eprintln!("[sidecar] failed to write PID file: {e}");
    }
}

/// Remove the PID file from the workspace directory.
pub fn remove_pid(workspace: &str) {
    let pid_path = Path::new(workspace).join(PID_FILE);
    let _ = std::fs::remove_file(&pid_path);
}

/// Spawn the Python sidecar and return (child, port).
///
/// Kills any stale sidecar first, writes a PID file for the new one.
pub async fn spawn_sidecar(workspace: &str, python: &str) -> Result<(tokio::process::Child, u16), String> {
    kill_stale(workspace);

    let mut child = Command::new(python)
        .args(["-m", "sigil_atlas.serve", "--workspace", workspace])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("Failed to spawn sidecar: {e}"))?;

    // Record PID so next launch can clean up if we crash
    if let Some(pid) = child.id() {
        write_pid(workspace, pid);
    }

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
