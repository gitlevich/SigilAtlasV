/**
 * Native macOS application menu — every user-invokable affordance is
 * available here, with a keyboard shortcut. Menu items dispatch through
 * `menu-actions.ts` so the same code runs whether the user picks a menu
 * item, presses a shortcut, or clicks a button in the side panel.
 *
 * Only active in Tauri (proper bundle). In browser dev mode, the panels
 * remain the only path.
 */

import * as A from "../menu-actions";

export async function initMenu(): Promise<void> {
  try {
    const { Menu, Submenu, MenuItem, PredefinedMenuItem } =
      await import("@tauri-apps/api/menu");

    // ── Sigil Atlas (App menu) ────────────────────────────────────────
    const appSubmenu = await Submenu.new({
      text: "Sigil Atlas",
      items: [
        await PredefinedMenuItem.new({ item: { About: { name: "Sigil Atlas" } } }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await MenuItem.new({
          id: "settings",
          text: "Settings\u2026",
          accelerator: "CmdOrCtrl+,",
          action: () => A.actOpenSettings(),
        }),
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

    // ── File ──────────────────────────────────────────────────────────
    const fileSubmenu = await Submenu.new({
      text: "File",
      items: [
        await MenuItem.new({
          id: "open-sigil",
          text: "Open Sigil\u2026",
          accelerator: "CmdOrCtrl+O",
          action: () => A.actOpenCollage(),
        }),
        await MenuItem.new({
          id: "save-sigil",
          text: "Save Sigil\u2026",
          accelerator: "CmdOrCtrl+S",
          action: () => A.actSaveCollage(),
        }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await MenuItem.new({
          id: "import",
          text: "Import Photos\u2026",
          accelerator: "CmdOrCtrl+Shift+I",
          action: () => A.actImportPhotos().catch((e) => console.error("[import]", e)),
        }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await PredefinedMenuItem.new({ item: "CloseWindow" }),
      ],
    });

    // ── Edit ──────────────────────────────────────────────────────────
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
        await PredefinedMenuItem.new({ item: "Separator" }),
        await MenuItem.new({
          id: "find",
          text: "Find\u2026",
          accelerator: "CmdOrCtrl+F",
          action: () => A.actFocusAttractInput(),
        }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await MenuItem.new({
          // Escape is the natural keystroke but isn't bound here as the
          // menu accelerator — that would intercept Escape globally and
          // break the @Lightbox's own Escape handling. The window-level
          // keydown in main.ts releases the target on Escape when no
          // other context owns the keystroke.
          id: "release-target",
          text: "Release Target Image",
          action: () => A.actReleaseTarget(),
        }),
        await MenuItem.new({
          id: "clear-attractors",
          text: "Clear All Attractors",
          accelerator: "CmdOrCtrl+Shift+Backspace",
          action: () => A.actClearAttractors(),
        }),
        await MenuItem.new({
          id: "clear-filters",
          text: "Clear All Filters",
          accelerator: "CmdOrCtrl+Alt+Shift+Backspace",
          action: () => A.actClearAllFilters(),
        }),
      ],
    });

    // ── View ──────────────────────────────────────────────────────────
    const viewSubmenu = await Submenu.new({
      text: "View",
      items: [
        await MenuItem.new({
          id: "zoom-in",
          text: "Zoom In",
          accelerator: "CmdOrCtrl+=",
          action: () => A.actZoomIn(),
        }),
        await MenuItem.new({
          id: "zoom-out",
          text: "Zoom Out",
          accelerator: "CmdOrCtrl+-",
          action: () => A.actZoomOut(),
        }),
        await MenuItem.new({
          id: "frame-all",
          text: "Frame All",
          accelerator: "CmdOrCtrl+0",
          action: () => A.actFrameAll(),
        }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await MenuItem.new({
          id: "mode-spacelike",
          text: "Spacelike",
          accelerator: "CmdOrCtrl+1",
          action: () => A.actModeSpacelike(),
        }),
        await MenuItem.new({
          id: "mode-timelike",
          text: "Timelike",
          accelerator: "CmdOrCtrl+2",
          action: () => A.actModeTimelike(),
        }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await MenuItem.new({
          id: "layer-photos",
          text: "Show Photos",
          accelerator: "CmdOrCtrl+Shift+P",
          action: () => A.actToggleLayerPhotos(),
        }),
        await MenuItem.new({
          id: "layer-neighborhoods",
          text: "Show Neighborhoods",
          accelerator: "CmdOrCtrl+Shift+N",
          action: () => A.actToggleLayerNeighborhoods(),
        }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await MenuItem.new({
          id: "toggle-sidebar",
          text: "Toggle Sidebar",
          accelerator: "CmdOrCtrl+\\",
          action: () => A.actToggleSidebar(),
        }),
      ],
    });

    // ── Image ─────────────────────────────────────────────────────────
    const imageSubmenu = await Submenu.new({
      text: "Image",
      items: [
        await MenuItem.new({
          id: "open-lightbox",
          text: "Open in Lightbox",
          accelerator: "CmdOrCtrl+L",
          action: () => A.actOpenLightbox(),
        }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await MenuItem.new({
          id: "set-target-centre",
          text: "Set Target to Centred Image",
          accelerator: "CmdOrCtrl+T",
          action: () => A.actSetTargetToCenter(),
        }),
        await MenuItem.new({
          id: "release-target-2",
          text: "Release Target Image",
          accelerator: "CmdOrCtrl+Shift+T",
          action: () => A.actReleaseTarget(),
        }),
      ],
    });

    // ── Tools ─────────────────────────────────────────────────────────
    const toolsSubmenu = await Submenu.new({
      text: "Tools",
      items: [
        await MenuItem.new({
          id: "embed-missing",
          text: "Embed Missing Models",
          action: () => A.actEmbedMissing().catch((e) => console.error("[embed-missing]", e)),
        }),
        await MenuItem.new({
          id: "pixel-features",
          text: "Recompute Pixel Features",
          action: () =>
            A.actRecomputePixelFeatures().catch((e) => console.error("[pixel-features]", e)),
        }),
        await MenuItem.new({
          id: "regenerate-previews",
          text: "Regenerate Previews",
          action: () =>
            A.actRegeneratePreviews().catch((e) => console.error("[regenerate-previews]", e)),
        }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await MenuItem.new({
          id: "nuke-corpus",
          text: "Nuke the Corpus\u2026",
          action: () => A.actNukeCorpus().catch((e) => console.error("[nuke]", e)),
        }),
      ],
    });

    // ── Window (standard Mac) ─────────────────────────────────────────
    const windowSubmenu = await Submenu.new({
      text: "Window",
      items: [
        await PredefinedMenuItem.new({ item: "Minimize" }),
        await PredefinedMenuItem.new({ item: "Maximize" }),
        await PredefinedMenuItem.new({ item: "Separator" }),
        await PredefinedMenuItem.new({ item: "Fullscreen" }),
      ],
    });

    const menu = await Menu.new({
      items: [
        appSubmenu,
        fileSubmenu,
        editSubmenu,
        viewSubmenu,
        imageSubmenu,
        toolsSubmenu,
        windowSubmenu,
      ],
    });
    await menu.setAsAppMenu();
  } catch {
    // Not in Tauri — no native menu
  }
}
