// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod sidecar;

use std::sync::Arc;
use tokio::sync::Mutex;

pub struct SidecarState {
    pub port: Arc<Mutex<Option<u16>>>,
    pub process: Arc<Mutex<Option<tokio::process::Child>>>,
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState {
            port: Arc::new(Mutex::new(None)),
            process: Arc::new(Mutex::new(None)),
        })
        .invoke_handler(tauri::generate_handler![
            commands::start_sidecar,
            commands::get_sidecar_port,
            commands::proxy_get,
            commands::proxy_post,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
