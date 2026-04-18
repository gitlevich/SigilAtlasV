// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod sidecar;

use std::sync::{Arc, Mutex as StdMutex};
use tauri::{Emitter, Manager};
use tokio::sync::Mutex;

pub struct SidecarState {
    pub port: Arc<Mutex<Option<u16>>>,
    pub process: Arc<Mutex<Option<tokio::process::Child>>>,
    pub workspace: Arc<Mutex<Option<String>>>,
}

fn main() {
    // Buffers any `.sigil` URL the OS asks us to open before the webview is
    // ready to receive events. `RunEvent::Opened` can fire on cold launch,
    // before the JS listener attaches; we flush these on window-ready.
    let pending_open: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
    let pending_for_run = pending_open.clone();
    let pending_for_setup = pending_open.clone();

    let app = tauri::Builder::default()
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
            commands::drain_pending_opens,
            commands::list_sigils,
            commands::delete_sigil,
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::Destroyed = event {
                let state = window.state::<SidecarState>();
                // Block briefly to kill the sidecar and clean up the PID file.
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();
                rt.block_on(async {
                    if let Some(mut child) = state.process.lock().await.take() {
                        let _ = child.kill().await;
                    }
                    if let Some(ws) = state.workspace.lock().await.as_deref() {
                        sidecar::remove_pid(ws);
                    }
                });
            }
        })
        .setup(move |app| {
            app.manage(commands::PendingOpens(pending_for_setup));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(move |app_handle, event| {
        if let tauri::RunEvent::Opened { urls } = event {
            for url in urls {
                let path = if url.scheme() == "file" {
                    url.to_file_path()
                        .ok()
                        .and_then(|p| p.to_str().map(String::from))
                } else {
                    Some(url.to_string())
                };
                let Some(path) = path else { continue };

                // Try to deliver immediately; if the webview isn't up, buffer.
                if let Some(window) = app_handle.get_webview_window("main") {
                    let _ = window.emit("collage-open", path.clone());
                }
                if let Ok(mut buf) = pending_for_run.lock() {
                    buf.push(path);
                }
            }
        }
    });
}
