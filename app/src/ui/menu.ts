/**
 * Native application menu — File > Import... (Cmd+I), Tools > Nuke the Corpus...
 *
 * Only active in Tauri. In browser dev mode, the Import section
 * in controls.ts serves as fallback.
 */

import * as api from "../api";
import { startImport } from "../import";

export async function initMenu(): Promise<void> {
  try {
    const { Menu, Submenu, MenuItem, PredefinedMenuItem } = await import(
      "@tauri-apps/api/menu"
    );
    const { open, confirm } = await import("@tauri-apps/plugin-dialog");

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
          action: async () => {
            const selected = await open({ directory: true, title: "Choose source folder" });
            if (typeof selected === "string") {
              startImport(selected).catch((e) => console.error("[import]", e));
            }
          },
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

    const toolsSubmenu = await Submenu.new({
      text: "Tools",
      items: [
        await MenuItem.new({
          id: "nuke-corpus",
          text: "Nuke the Corpus...",
          action: async () => {
            const yes = await confirm(
              "This will permanently delete all images, embeddings, and metadata from the corpus. This cannot be undone.",
              { title: "Nuke the Corpus" },
            );
            if (yes) {
              api.nukeCorpus().catch((e) => console.error("[nuke]", e));
            }
          },
        }),
      ],
    });

    const menu = await Menu.new({
      items: [appSubmenu, fileSubmenu, editSubmenu, toolsSubmenu],
    });
    await menu.setAsAppMenu();
  } catch {
    // Not in Tauri — no native menu
  }
}
