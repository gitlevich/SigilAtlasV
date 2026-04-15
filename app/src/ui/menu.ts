/**
 * Native application menu — File > Import... with Cmd+I.
 *
 * Only active in Tauri. In browser dev mode, the Import section
 * in controls.ts serves as fallback.
 */

import * as api from "../api";
import { startPolling } from "./status-bar";

export async function initMenu(): Promise<void> {
  try {
    const { Menu, Submenu, MenuItem, PredefinedMenuItem } = await import(
      "@tauri-apps/api/menu"
    );
    const { open } = await import("@tauri-apps/plugin-dialog");

    const appSubmenu = await Submenu.new({
      text: "Sigil Atlas",
      items: [
        await PredefinedMenuItem.new({ item: { About: { name: "Sigil Atlas" } } }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await PredefinedMenuItem.new({ item: "Services" }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await PredefinedMenuItem.new({ item: "Hide" }),
        await PredefinedMenuItem.new({ item: "HideOthers" }),
        await PredefinedMenuItem.new({ item: "ShowAll" }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await PredefinedMenuItem.new({ item: "Quit" }),
      ],
    });

    const fileSubmenu = await Submenu.new({
      text: "File",
      items: [
        await MenuItem.new({
          id: "import",
          text: "Import...",
          accelerator: "CmdOrCtrl+I",
          action: () => handleImport(open),
        }),
      ],
    });

    const editSubmenu = await Submenu.new({
      text: "Edit",
      items: [
        await PredefinedMenuItem.new({ item: "Undo" }),
        await PredefinedMenuItem.new({ item: "Redo" }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await PredefinedMenuItem.new({ item: "Cut" }),
        await PredefinedMenuItem.new({ item: "Copy" }),
        await PredefinedMenuItem.new({ item: "Paste" }),
        await PredefinedMenuItem.new({ item: "SelectAll" }),
      ],
    });

    const menu = await Menu.new({
      items: [appSubmenu, fileSubmenu, editSubmenu],
    });
    await menu.setAsAppMenu();
  } catch {
    // Not in Tauri — no native menu, panel fallback is enough
  }
}

type DialogOpen = (opts: { directory: boolean; title: string }) => Promise<string | string[] | null>;

async function handleImport(open: DialogOpen): Promise<void> {
  const selected = await open({ directory: true, title: "Choose source folder" });
  if (typeof selected !== "string") return;

  try {
    const res = await api.startImport(selected);
    if (res.status === "started") {
      startPolling();
    }
  } catch (e) {
    console.error("Import failed to start:", e);
  }
}
