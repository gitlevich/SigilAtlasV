// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod sidecar;

use std::sync::Arc;
use tauri::Manager;
use tokio::sync::Mutex;

pub struct SidecarState {
    pub port: Arc<Mutex<Option<u16>>>,
    pub process: Arc<Mutex<Option<tokio::process::Child>>>,
    pub workspace: Arc<Mutex<Option<String>>>,
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(SidecarState {
            port: Arc::new(Mutex::new(None)),
            process: Arc::new(Mutex::new(None)),
            workspace: Arc::new(Mutex::new(None)),
        })
        .invoke_handler(tauri::generate_handler![
            commands::start_sidecar,
            commands::get_sidecar_port,
            commands::proxy_get,
            commands::proxy_post,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.state::<SidecarState>();
                // Block briefly to kill the sidecar and clean up the PID file.
                // This runs on the main thread during window close — acceptable
                // since it's just a SIGTERM + file delete.
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    // Kill the child process explicitly
                    if let Some(mut child) = state.process.lock().await.take() {
                        let _ = child.kill().await;
                    }
                    // Remove PID file
                    if let Some(ws) = state.workspace.lock().await.as_deref() {
                        sidecar::remove_pid(ws);
                    }
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
