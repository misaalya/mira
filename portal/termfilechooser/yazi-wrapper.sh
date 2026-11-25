#!/usr/bin/env sh
# ╭──────────────────────────────────────────────────────────────────────────╮
# │  Misa — wrapper yazi untuk xdg-desktop-portal-termfilechooser            │
# ╰──────────────────────────────────────────────────────────────────────────╯
#
# Dipakai lewat `cmd=yazi-wrapper.sh` di
# $XDG_CONFIG_HOME/xdg-desktop-portal-termfilechooser/config. Portal mencari
# wrapper di folder config user LEBIH DULU daripada /usr/share/…, jadi berkas
# ini yang dipakai (bukan yazi-wrapper.sh bawaan paket).
#
# Argumen (lihat `man 5 xdg-desktop-portal-termfilechooser`):
#   $1 multiple · $2 directory · $3 save · $4 path · $5 out · $6 debug
# sisanya adalah TERMCMD yang sudah dipecah jadi kata (ghostty … -e).
#
# ── KENAPA TIDAK MEMAKAI WRAPPER BAWAAN ────────────────────────────────────
#
# Dengan `create_help_file=1` (bawaan) portal menulis TEKS INSTRUKSI ke file
# tujuan sebelum yazi dibuka, lalu yazi dipakai untuk me-rename file itu dan
# "open" sebagai tanda simpan. Masalahnya: kalau dialog dibatalkan setelah
# file di-rename (`Q`, Ctrl+C, atau jendela ghostty ditutup), portal gagal
# membaca /tmp/termfilechooser-<uid>.portal, lalu hanya menghapus path LAMA
# yang sudah tidak ada. Hasilnya teks instruksi tertinggal di folder tujuan —
# mis. unduhan browser jadi `~/gambar.jpeg` berisi
# "xdg-desktop-portal-termfilechooser instructions", padahal gambarnya tidak
# pernah ditulis (portal membalas error, unduhan dibatalkan).
#
# Di sini config memakai `create_help_file=0` (portal tidak menulis apa pun)
# dan wrapper membuat file tujuan KOSONG sebagai gantinya:
#
#   · alur tetap sama: yazi terbuka dengan file tujuan, rename (`r`) kalau
#     perlu, lalu Enter (atau `q`) untuk menyimpan;
#   · kalau dibatalkan, file itu 0 byte (bukan teks menyesatkan) dan wrapper
#     menghapusnya lagi — termasuk kalau sudah di-rename, dicari lewat inode;
#   · kalau wrapper-nya sendiri dibunuh portal (pembatalan keras), penjaga
#     kecil di bawah yang membersihkan file 0 byte itu.
#
# Mode lain (buka file, pilih banyak file, pilih folder) sama seperti bawaan.
set -u

multiple="$1"
directory="$2"
save="$3"
path="$4"
out="$5"
debug="$6"
shift 6 # sisanya TERMCMD

[ "$debug" = 1 ] && set -x

# ── placeholder mode simpan ────────────────────────────────────────────────
# Hanya kalau path tujuan belum ada. Kalau sudah ada (menimpa file lama),
# jangan disentuh sama sekali.
placeholder=""
inode=""
if [ "$save" = 1 ] && [ -n "$path" ] && [ ! -e "$path" ]; then
    if ( : >"$path" ) 2>/dev/null; then
        placeholder="$path"
        inode=$(stat -c %i -- "$path" 2>/dev/null || true)
    fi
fi

# ── penjaga placeholder (kalau wrapper dimatikan paksa) ────────────────────
# Kalau portal membatalkan permintaan (mis. unduhan dibatalkan di browser) ia
# bisa MEMBUNUH wrapper ini, jadi baris pembersihan di bawah tak pernah jalan
# dan file 0 byte tadi tertinggal di folder tujuan — terlihat persis seperti
# hasil unduhan yang kosong. Penjaga ini menunggu PID wrapper hilang, lalu
# menghapus file itu HANYA kalau ia masih placeholder yang sama (inode cocok)
# dan masih 0 byte, jadi file hasil simpan yang sudah terisi tidak tersentuh.
# Aman terhadap SIGKILL karena penjaga adalah proses terpisah.
watch=""
if [ -n "$placeholder" ] && [ -n "$inode" ]; then
    (
        while kill -0 "$$" 2>/dev/null; do sleep 1; done
        if [ -e "$placeholder" ] &&
            [ "$(stat -c %i -- "$placeholder" 2>/dev/null)" = "$inode" ] &&
            [ ! -s "$placeholder" ]; then
            rm -f -- "$placeholder"
        fi
    ) </dev/null >/dev/null 2>&1 &
    watch=$!
fi

# ── jalankan terminal + yazi ───────────────────────────────────────────────
if [ "$directory" = 1 ]; then
    set -- "$@" yazi --chooser-file="$out" --cwd-file="$out.1" "$path"
else
    set -- "$@" yazi --chooser-file="$out" "$path"
fi
"$@"
status=$?

# Pilih folder: kalau yazi keluar dari sebuah folder, pilihannya ditulis ke
# $out.1; jadikan itu hasil utama (perilaku bawaan).
if [ "$directory" = 1 ]; then
    if [ ! -s "$out" ] && [ -s "$out.1" ]; then
        cat "$out.1" >"$out"
        rm -f -- "$out.1"
    else
        rm -f -- "$out.1"
    fi
fi

# ── buang placeholder kalau simpan dibatalkan ──────────────────────────────
# yazi hanya menulis $out saat file dipilih/dibuka. Kosong = batal, jadi file
# kosong tadi tidak boleh tertinggal di folder tujuan.
if [ "$save" = 1 ] && [ -n "$placeholder" ] && [ ! -s "$out" ]; then
    if [ -e "$placeholder" ] && [ -n "$inode" ] &&
        [ "$(stat -c %i -- "$placeholder" 2>/dev/null)" = "$inode" ]; then
        rm -f -- "$placeholder"
    elif [ -n "$inode" ]; then
        # Sudah di-rename: cari file dengan inode yang sama di folder itu
        # (termasuk subfoldernya, karena yazi bisa memindahkan file).
        found=$(find "$(dirname -- "$placeholder")" -maxdepth 3 -xdev \
            -type f -inum "$inode" -print 2>/dev/null | head -n 1)
        [ -n "$found" ] && rm -f -- "$found"
    fi
fi

# Wrapper keluar normal: pengawas tak diperlukan lagi, matikan supaya tidak
# sempat menghapus apa pun setelah ini.
[ -n "$watch" ] && kill "$watch" 2>/dev/null

exit "$status"
