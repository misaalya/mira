# AGENTS.md — Mira (BSPWM / Xorg · Debian 13)

Repo ini adalah proyek **Mira** (oleh **Misaalya**) berbasis **BSPWM / Xorg**
untuk **Debian 13 (trixie)**. Kerangka skrip sesi modular (`Theme.sh`,
`WallSelect`, `RiceSelector`, `BarSelector`, `SoftReload`, `HideBar`, dll.)
diadaptasi secara ramping tanpa bloat dari `gh0stzk/dotfiles` (GPL-3.0) dengan
atribusi penuh, sementara desain visual **Mira**, topbar, panel, dan seluruh
daemon/UI native Rust & C (`src/*`) adalah kode asli proyek ini.

---

## 1. Struktur Repo

```text
src/
  misa-scroll/      Daemon layout kolom bergulir BSPWM (Rust, 0 crate)
  misa-clip/        Daemon riwayat clipboard teks + PNG (Rust, 0 crate)
  misa-backdrop/    Backdrop blur/dim + passive Escape grab (Rust, 0 crate)
  misa-wall/        Engine wallpaper X11 SIMD AVX2/SSE2 + MIT-SHM (Rust)
  misa-ui/          Popup & overlay GTK3 dalam C (NotifPanel, ClipPanel,
                    WifiPanel, BluetoothPanel, SoundPanel, MusicPanel,
                    MediaPillUI, CardPicker, WindowSwitch, WaitCompositor)
session/
  bspwmrc           Autostart sesi BSPWM (display-scoped, mendukung $MIRA_CONFIG_DIR)
  session.sh        Entry point SDDM ("Mira (BSPWM)")
  themes/mira/      Tema bawaan: Mira (theme.conf + walls/)
  bars/             Varian topbar Polybar monokrom (mira, mira-float, mira-islands,
                    mira-line, mira-minimal)
  config/
    sxhkdrc         Shortcut bersih khusus Mira
    system.ini      Variabel hardware (diisi otomatis oleh SetSysVars)
    modules/        Generator tema (00-picom .. 07-gtk; tanpa dunst/legacy bloat)
    picom/          picom.conf, picom-rules.conf, picom-animations.conf
    rofi/           launcher.rasi, BarMenu.rasi, PowerMenu.rasi, keybinds.rasi, Hide.rasi
    *.conf          scroll.conf, clip.conf, wallselect.conf, windowswitch.conf
  bin/              Skrip kontrol panel & sesi (semua menghormati $MIRA_CONFIG_DIR)
assets/fonts/       Font ikon (Material Design Icons Desktop)
portal/             Konfigurasi xdg-desktop-portal + termfilechooser (yazi)
tools/
  build.sh          Build seluruh komponen Rust & C ke ./build/bin
  dev-session.sh    Menjalankan sesi penuh terisolasi di Xephyr (:9) + dbus-run-session
  deploy.sh         Deploy aman ke ~/.config/bspwm (validasi picom dulu)
  setup.sh          (sudo) Paket Debian 13 + touchpad + entri SDDM
```

---

## 2. Aturan Wajib

1. **Jangan Ganggu Sesi Live (`:0`) & Deploy Otomatis:**
   - Uji seluruh perubahan di sandbox Xephyr terisolasi (`./tools/dev-session.sh`),
     yang mengisolasi `MIRA_CONFIG_DIR`, `MIRA_SANDBOX=1`, `XDG_RUNTIME_DIR`,
     `XDG_CACHE_HOME`, `BSPWM_SOCKET`, dan `dbus-run-session`.
   - Setelah perubahan selesai dan terverifikasi, selalu jalankan `./tools/deploy.sh`
     untuk menerapkan perubahan ke `~/.config/bspwm`.
   - Tutup Xephyr setelah selesai; kill proses hanya per PID.
2. **Privasi & Kebersihan Git:**
   - **Tanpa path hardcode** (`/home/<user>`) — gunakan `$MIRA_CONFIG_DIR`,
     `$HOME`, `$XDG_*`, atau path relatif.
   - Tanpa data pribadi (SSID, IP/MAC, riwayat clipboard/notifikasi, cache).
   - Commit ke git hanya jika diminta user.
3. **Lisensi & Atribusi:**
   - Lisensi repo ini **GPL-3.0** (`LICENSE`). Pertahankan header atribusi
     `gh0stzk/dotfiles` pada skrip shell yang diadaptasi dari sana.
4. **Bahasa:**
   - User menulis dalam Bahasa Indonesia; jawab dalam Bahasa Indonesia.
