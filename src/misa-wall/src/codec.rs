//! Crate-less Planar-Delta BitPack + 64K-hash LZ4 block codec, 16x9 tile dirty
//! bitmask, content-hashed LRU cache, and zero-PSS `pread` streaming `.mwa` v3
//! animation container for `misa-wall`.

use std::ffi::c_int;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const POSIX_FADV_SEQUENTIAL: c_int = 2;
const POSIX_FADV_DONTNEED: c_int = 4;

extern "C" {
    fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) -> c_int;
}

pub const MWA_MAGIC: [u8; 8] = *b"MWA3\0\0\0\0";
pub const MWA_VERSION: u32 = 3;
pub const HEADER_SIZE: usize = 64;
pub const ENTRY_SIZE: usize = 44;

// Coarse dirty tile grid: 16x9 (40x40 cells per tile on 640x360 = 144 tiles).
pub const TILE_COLS: usize = 16;
pub const TILE_ROWS: usize = 9;
pub const CACHE_KEEP_NEWEST: usize = 12;

#[derive(Clone, Copy, Debug)]
pub struct FrameEntry {
    pub data_offset: u64,
    pub comp_len: u32,
    pub raw_len: u32,
    pub dirty_x0: u16,
    pub dirty_y0: u16,
    pub dirty_x1: u16,
    pub dirty_y1: u16,
    pub tile_mask: [u16; TILE_ROWS],
}

impl FrameEntry {
    pub fn empty() -> Self {
        Self {
            data_offset: 0,
            comp_len: 0,
            raw_len: 0,
            dirty_x0: 0,
            dirty_y0: 0,
            dirty_x1: 0,
            dirty_y1: 0,
            tile_mask: [0u16; TILE_ROWS],
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.comp_len == 0 || self.dirty_x1 <= self.dirty_x0 || self.dirty_y1 <= self.dirty_y0
    }
}

/// Merge dirty `16x9` tiles into a small set of tight cell-coordinate rectangles
/// `(cx0, cy0, cx1, cy1)` clamped to `(0..cw, 0..ch)`. Adjacent dirty tiles in the
/// same row merge horizontally, and consecutive rows with identical spans merge
/// vertically. If the tile coverage exceeds 65% or fragments into >16 rects,
/// returns the single tight bounding box to keep X11 request overhead minimal.
pub fn merge_tile_rects(
    tile_mask: &[u16; TILE_ROWS],
    bbox: (u16, u16, u16, u16),
    cw: i32,
    ch: i32,
    out: &mut Vec<(i32, i32, i32, i32)>,
) {
    out.clear();
    let bx0 = (bbox.0 as i32).clamp(0, cw);
    let by0 = (bbox.1 as i32).clamp(0, ch);
    let bx1 = (bbox.2 as i32).clamp(bx0, cw);
    let by1 = (bbox.3 as i32).clamp(by0, ch);
    if bx1 <= bx0 || by1 <= by0 {
        return;
    }

    let mut dirty_tiles = 0u32;
    for &row in tile_mask {
        dirty_tiles += row.count_ones();
    }
    if dirty_tiles == 0 {
        out.push((bx0, by0, bx1, by1));
        return;
    }
    if dirty_tiles >= ((TILE_COLS * TILE_ROWS) as u32 * 65) / 100 {
        out.push((bx0, by0, bx1, by1));
        return;
    }

    // Build horizontal runs per tile row and merge vertically with previous row.
    let mut rects: Vec<(usize, usize, usize, usize)> = Vec::with_capacity(16); // (tx0, ty0, tx1, ty1)
    for ty in 0..TILE_ROWS {
        let mut m = tile_mask[ty];
        while m != 0 {
            let tx0 = m.trailing_zeros() as usize;
            let shifted = m >> tx0;
            let run = (!shifted).trailing_zeros() as usize;
            let tx1 = (tx0 + run).min(TILE_COLS);
            let clear_mask: u16 = if tx1 - tx0 >= 16 {
                u16::MAX
            } else {
                (((1u32 << (tx1 - tx0)) - 1) << tx0) as u16
            };
            m &= !clear_mask;

            // Try extending an existing rect ending on row `ty` with matching [tx0..tx1).
            let mut merged = false;
            for r in rects.iter_mut().rev() {
                if r.3 == ty && r.0 == tx0 && r.2 == tx1 {
                    r.3 = ty + 1;
                    merged = true;
                    break;
                }
            }
            if !merged {
                rects.push((tx0, ty, tx1, ty + 1));
            }
        }
    }

    if rects.is_empty() || rects.len() > 16 {
        out.push((bx0, by0, bx1, by1));
        return;
    }

    for (tx0, ty0, tx1, ty1) in rects {
        let x0 = ((tx0 as i32 * cw) / TILE_COLS as i32).max(bx0);
        let y0 = ((ty0 as i32 * ch) / TILE_ROWS as i32).max(by0);
        let x1 = (((tx1 as i32 * cw) + TILE_COLS as i32 - 1) / TILE_COLS as i32).min(bx1);
        let y1 = (((ty1 as i32 * ch) + TILE_ROWS as i32 - 1) / TILE_ROWS as i32).min(by1);
        if x1 > x0 && y1 > y0 {
            out.push((x0, y0, x1, y1));
        }
    }
    if out.is_empty() {
        out.push((bx0, by0, bx1, by1));
    }
}

/// FNV-1a 64-bit hash for deterministic content-addressed cache keys.
pub struct Fnv64(u64);

impl Fnv64 {
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    #[inline]
    pub fn update(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h;
    }

    #[inline]
    pub fn update_u64(&mut self, v: u64) {
        self.update(&v.to_le_bytes());
    }

    #[inline]
    pub fn update_u32(&mut self, v: u32) {
        self.update(&v.to_le_bytes());
    }

    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// Compute a deterministic content-addressed cache key from exact image bytes +
/// exact shader bytes + cell grid dimensions & playback parameters. Independent
/// of file `mtime` so re-deploying unchanged assets never duplicates cache files!
pub fn compute_cache_hash(
    img_path: &Path,
    frag_path: &Path,
    cell_w: u32,
    cell_h: u32,
    fps: u32,
    frames: u32,
) -> Option<u64> {
    let img_bytes = fs::read(img_path).ok()?;
    let frag_bytes = fs::read(frag_path).ok()?;

    let mut h = Fnv64::new();
    h.update(b"MWA3-CONTENT-V3");
    h.update_u32(cell_w);
    h.update_u32(cell_h);
    h.update_u32(fps);
    h.update_u32(frames);
    h.update_u64(img_bytes.len() as u64);
    h.update(&img_bytes);
    h.update_u64(frag_bytes.len() as u64);
    h.update(&frag_bytes);
    Some(h.finish())
}

pub fn cache_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("mira").join("wall-anim");
        }
    }
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home).join(".cache").join("mira").join("wall-anim");
        }
    }
    PathBuf::from("/tmp/mira-wall-anim")
}

pub fn cache_path_for_hash(hash: u64) -> PathBuf {
    cache_dir().join(format!("{hash:016x}.mwa"))
}

pub fn clear_cache_dir() -> usize {
    let dir = cache_dir();
    let mut removed = 0usize;
    if let Ok(rd) = fs::read_dir(&dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("mwa") {
                if fs::remove_file(&p).is_ok() {
                    removed += 1;
                }
            }
        }
    }
    removed
}

/// Remove obsolete MWA2 files and keep at most `keep_newest` valid MWA3 files
/// by mtime (protecting `protect` if given) so `~/.cache/mira/wall-anim` stays tidy.
pub fn prune_cache_dir(keep_newest: usize, protect: Option<&Path>) {
    let dir = cache_dir();
    let Ok(rd) = fs::read_dir(&dir) else { return; };
    let mut v3_files: Vec<(SystemTime, PathBuf)> = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("mwa") {
            continue;
        }
        if protect.map(|pr| pr == p.as_path()).unwrap_or(false) {
            continue;
        }
        // Immediately purge legacy non-MWA3 cache files.
        let mut magic = [0u8; 8];
        let is_v3 = File::open(&p)
            .and_then(|f| f.read_exact_at(&mut magic, 0))
            .map(|_| magic == MWA_MAGIC)
            .unwrap_or(false);
        if !is_v3 {
            let _ = fs::remove_file(&p);
            continue;
        }
        let mt = e
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        v3_files.push((mt, p));
    }
    let limit = if protect.is_some() {
        keep_newest.saturating_sub(1)
    } else {
        keep_newest
    };
    if v3_files.len() <= limit {
        return;
    }
    v3_files.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, old) in v3_files.into_iter().skip(limit) {
        let _ = fs::remove_file(old);
    }
}

// ============================================================================
// Pure-Rust, zero-dependency standard LZ4 block compressor (64K hash) & decoder
// ============================================================================

const LZ4_HASH_LOG: usize = 14; // 16384 entries = 64 KB hash table (better ratio!)
const LZ4_HASH_SIZE: usize = 1 << LZ4_HASH_LOG;
const LZ4_MIN_MATCH: usize = 4;
const LZ4_MAX_DIST: usize = 65535;
const LZ4_MFLIMIT: usize = 12;
const LZ4_LAST_LITERALS: usize = 5;

#[inline(always)]
fn read_u32_le(src: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([src[i], src[i + 1], src[i + 2], src[i + 3]])
}

#[inline(always)]
fn lz4_hash(v: u32) -> usize {
    ((v.wrapping_mul(2654435761)) >> (32 - LZ4_HASH_LOG)) as usize
}

#[inline]
fn write_ext_len(out: &mut Vec<u8>, mut extra: usize) {
    while extra >= 255 {
        out.push(255);
        extra -= 255;
    }
    out.push(extra as u8);
}

/// Compress `src` using the standard LZ4 block format with a 64 KB hash table.
pub fn lz4_compress(src: &[u8]) -> Vec<u8> {
    let len = src.len();
    let mut out = Vec::with_capacity(len + (len >> 8) + 16);
    if len == 0 {
        return out;
    }
    if len < LZ4_MFLIMIT {
        let lit_tok = len.min(15) as u8;
        out.push(lit_tok << 4);
        if len >= 15 {
            write_ext_len(&mut out, len - 15);
        }
        out.extend_from_slice(src);
        return out;
    }

    let mut table = vec![u32::MAX; LZ4_HASH_SIZE];
    let mut anchor = 0usize;
    let mut ip = 0usize;
    let mflimit = len - LZ4_MFLIMIT;
    let matchlimit = len - LZ4_LAST_LITERALS;

    while ip <= mflimit {
        let cur = read_u32_le(src, ip);
        let h = lz4_hash(cur);
        let cand = table[h];
        table[h] = ip as u32;

        let mut matched = false;
        let mut ref_pos = 0usize;
        if cand != u32::MAX {
            let rp = cand as usize;
            let dist = ip - rp;
            if dist > 0 && dist <= LZ4_MAX_DIST && read_u32_le(src, rp) == cur {
                matched = true;
                ref_pos = rp;
            }
        }

        if !matched {
            let step = 1 + ((ip - anchor) >> 7);
            ip += step;
            continue;
        }

        while ip > anchor && ref_pos > 0 && src[ip - 1] == src[ref_pos - 1] {
            ip -= 1;
            ref_pos -= 1;
        }

        let mut ml = LZ4_MIN_MATCH;
        while ip + ml < matchlimit && src[ref_pos + ml] == src[ip + ml] {
            ml += 1;
        }

        let lit_len = ip - anchor;
        let ml_code = ml - LZ4_MIN_MATCH;
        let token = ((lit_len.min(15) as u8) << 4) | (ml_code.min(15) as u8);
        out.push(token);
        if lit_len >= 15 {
            write_ext_len(&mut out, lit_len - 15);
        }
        out.extend_from_slice(&src[anchor..ip]);

        let offset = (ip - ref_pos) as u16;
        out.extend_from_slice(&offset.to_le_bytes());

        if ml_code >= 15 {
            write_ext_len(&mut out, ml_code - 15);
        }

        ip += ml;
        anchor = ip;
        if ip <= mflimit {
            let p2 = ip - 2;
            table[lz4_hash(read_u32_le(src, p2))] = p2 as u32;
        }
    }

    let last_lit = len - anchor;
    let tok = (last_lit.min(15) as u8) << 4;
    out.push(tok);
    if last_lit >= 15 {
        write_ext_len(&mut out, last_lit - 15);
    }
    out.extend_from_slice(&src[anchor..len]);
    out
}

/// Decompress a standard LZ4 block into `dst` (which must have exact target size).
pub fn lz4_decompress(src: &[u8], dst: &mut [u8]) -> Result<usize, &'static str> {
    let slen = src.len();
    let dlen = dst.len();
    let mut sp = 0usize;
    let mut dp = 0usize;

    while sp < slen {
        let token = src[sp];
        sp += 1;

        let mut lit_len = (token >> 4) as usize;
        if lit_len == 15 {
            loop {
                if sp >= slen {
                    return Err("truncated literal length");
                }
                let b = src[sp] as usize;
                sp += 1;
                lit_len += b;
                if b != 255 {
                    break;
                }
            }
        }
        if sp + lit_len > slen || dp + lit_len > dlen {
            return Err("literal out of bounds");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(sp), dst.as_mut_ptr().add(dp), lit_len);
        }
        sp += lit_len;
        dp += lit_len;

        if sp >= slen {
            break;
        }
        if sp + 2 > slen {
            return Err("truncated match offset");
        }
        let offset = u16::from_le_bytes([src[sp], src[sp + 1]]) as usize;
        sp += 2;
        if offset == 0 || offset > dp {
            return Err("invalid match offset");
        }

        let mut match_len = (token & 0x0F) as usize + LZ4_MIN_MATCH;
        if (token & 0x0F) == 15 {
            loop {
                if sp >= slen {
                    return Err("truncated match length");
                }
                let b = src[sp] as usize;
                sp += 1;
                match_len += b;
                if b != 255 {
                    break;
                }
            }
        }
        if dp + match_len > dlen {
            return Err("match out of bounds");
        }

        let mut mp = dp - offset;
        if offset >= match_len {
            unsafe {
                let ptr = dst.as_mut_ptr();
                std::ptr::copy_nonoverlapping(ptr.add(mp), ptr.add(dp), match_len);
            }
            dp += match_len;
        } else if offset == 1 {
            let val = dst[mp];
            dst[dp..dp + match_len].fill(val);
            dp += match_len;
        } else {
            for _ in 0..match_len {
                dst[dp] = dst[mp];
                dp += 1;
                mp += 1;
            }
        }
    }

    Ok(dp)
}

// ============================================================================
// Planar BGR Base Frame + Planar-Delta BitPack (100% lossless, direct BGRX)
// ============================================================================

/// Encode base RGB888 frame (`w * h * 3`) into horizontally-predicted planar BGR
/// `[B_delta_plane | G_delta_plane | R_delta_plane]` for dramatically tighter LZ4!
pub fn encode_base_planar(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    let n = w * h;
    debug_assert_eq!(rgb.len(), n * 3);
    let mut out = vec![0u8; n * 3];
    let (b_plane, rest) = out.split_at_mut(n);
    let (g_plane, r_plane) = rest.split_at_mut(n);

    for y in 0..h {
        let row = y * w;
        let mut pb = 0u8;
        let mut pg = 0u8;
        let mut pr = 0u8;
        for x in 0..w {
            let p = row + x;
            let r = rgb[p * 3];
            let g = rgb[p * 3 + 1];
            let b = rgb[p * 3 + 2];
            b_plane[p] = b.wrapping_sub(pb);
            g_plane[p] = g.wrapping_sub(pg);
            r_plane[p] = r.wrapping_sub(pr);
            pb = b;
            pg = g;
            pr = r;
        }
    }
    out
}

/// Decode horizontally-predicted planar BGR directly into `dst_bgrx` (`w * h * 4`, byte 3 = 0).
pub fn decode_base_planar_bgrx(planar: &[u8], w: usize, h: usize, dst_bgrx: &mut [u8]) -> Result<(), &'static str> {
    let n = w * h;
    if planar.len() != n * 3 || dst_bgrx.len() < n * 4 {
        return Err("base planar size mismatch");
    }
    let b_plane = &planar[0..n];
    let g_plane = &planar[n..n * 2];
    let r_plane = &planar[n * 2..n * 3];
    let dst = dst_bgrx.as_mut_ptr();

    unsafe {
        for y in 0..h {
            let row = y * w;
            let mut pb = 0u8;
            let mut pg = 0u8;
            let mut pr = 0u8;
            for x in 0..w {
                let p = row + x;
                pb = pb.wrapping_add(*b_plane.get_unchecked(p));
                pg = pg.wrapping_add(*g_plane.get_unchecked(p));
                pr = pr.wrapping_add(*r_plane.get_unchecked(p));
                let dp = dst.add(p * 4);
                *dp = pb;
                *dp.add(1) = pg;
                *dp.add(2) = pr;
                *dp.add(3) = 0;
            }
        }
    }
    Ok(())
}

#[inline(always)]
unsafe fn count_equal_rgb(a: &[u8], b: &[u8], mut i: usize) -> usize {
    let len = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut eq = 0usize;

    while i + 8 <= len {
        let va = (ap.add(i) as *const u64).read_unaligned();
        let vb = (bp.add(i) as *const u64).read_unaligned();
        let x = va ^ vb;
        if x != 0 {
            eq += (x.trailing_zeros() as usize) / 24;
            return eq;
        }
        eq += 2;
        i += 6;
    }

    while i + 3 <= len {
        if *ap.add(i) != *bp.add(i)
            || *ap.add(i + 1) != *bp.add(i + 1)
            || *ap.add(i + 2) != *bp.add(i + 2)
        {
            break;
        }
        eq += 1;
        i += 3;
    }
    eq
}

#[inline(always)]
unsafe fn count_diff_rgb(a: &[u8], b: &[u8], mut i: usize) -> usize {
    let len = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut diff = 0usize;

    while i + 3 <= len {
        if *ap.add(i) == *bp.add(i)
            && *ap.add(i + 1) == *bp.add(i + 1)
            && *ap.add(i + 2) == *bp.add(i + 2)
        {
            break;
        }
        diff += 1;
        i += 3;
    }
    diff
}

#[inline(always)]
fn mark_tiles_span(
    tile_mask: &mut [u16; TILE_ROWS],
    p0: usize,
    p1: usize,
    w: usize,
    h: usize,
) {
    let y0 = p0 / w;
    let y1 = (p1 - 1) / w;
    let ty0 = (y0 * TILE_ROWS) / h;
    let ty1 = ((y1 * TILE_ROWS) / h).min(TILE_ROWS - 1);

    if y0 == y1 {
        let x0 = p0 - y0 * w;
        let x1 = p1 - y0 * w - 1;
        let tx0 = (x0 * TILE_COLS) / w;
        let tx1 = ((x1 * TILE_COLS) / w).min(TILE_COLS - 1);
        let span = tx1 - tx0 + 1;
        let bits: u16 = if span >= 16 {
            u16::MAX
        } else {
            (((1u32 << span) - 1) << tx0) as u16
        };
        tile_mask[ty0] |= bits;
    } else {
        // First partial row: [x0..w)
        let x0 = p0 - y0 * w;
        let tx0 = (x0 * TILE_COLS) / w;
        let bits0: u16 = u16::MAX << tx0;
        tile_mask[ty0] |= bits0;

        // Last partial row: [0..x1]
        let x1 = (p1 - 1) - y1 * w;
        let tx1 = ((x1 * TILE_COLS) / w).min(TILE_COLS - 1);
        let span1 = tx1 + 1;
        let bits1: u16 = if span1 >= 16 {
            u16::MAX
        } else {
            ((1u32 << span1) - 1) as u16
        };
        tile_mask[ty1] |= bits1;

        // Any full rows between y0 and y1 cover all tile columns.
        if y1 > y0 + 1 {
            let mid_y0 = ((y0 + 1) * TILE_ROWS) / h;
            let mid_y1 = (((y1 - 1) * TILE_ROWS) / h).min(TILE_ROWS - 1);
            for ty in mid_y0..=mid_y1 {
                tile_mask[ty] = u16::MAX;
            }
        }
    }
}

/// Packed delta metadata returned by `pack_planar_delta`.
pub struct PackedMeta {
    pub x0: u16,
    pub y0: u16,
    pub x1: u16,
    pub y1: u16,
    pub tile_mask: [u16; TILE_ROWS],
}

/// Encode pixel delta `prev -> cur` (both `w * h * 3` RGB888) into planar-delta
/// BitPack layout:
///   `[runs_len: u32][n_px: u32][runs_stream][dB_stream][dG_stream][dR_stream]`
/// Separating control runs and per-channel wrapping deltas makes LZ4 compress
/// ~2x tighter while remaining 100% lossless and enabling direct BGRX unpack!
pub fn pack_planar_delta(
    prev: &[u8],
    cur: &[u8],
    w: usize,
    h: usize,
    out: &mut Vec<u8>,
) -> Option<PackedMeta> {
    debug_assert_eq!(prev.len(), cur.len());
    debug_assert_eq!(prev.len(), w * h * 3);
    out.clear();

    let mut runs: Vec<u8> = Vec::with_capacity(4096);
    let mut db: Vec<u8> = Vec::with_capacity(16384);
    let mut dg: Vec<u8> = Vec::with_capacity(16384);
    let mut dr: Vec<u8> = Vec::with_capacity(16384);

    let mut min_x = w;
    let mut min_y = h;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut tile_mask = [0u16; TILE_ROWS];

    let mut i = 0usize;
    let len = prev.len();
    while i < len {
        let eq = unsafe { count_equal_rgb(prev, cur, i) };
        i += eq * 3;
        if i >= len {
            break;
        }

        let start = i;
        let diffs = unsafe { count_diff_rgb(prev, cur, i) };
        i += diffs * 3;

        let p0 = start / 3;
        let p1 = i / 3;
        mark_tiles_span(&mut tile_mask, p0, p1, w, h);

        let y0 = p0 / w;
        let y1 = (p1 - 1) / w;
        if y0 < min_y {
            min_y = y0;
        }
        if y1 + 1 > max_y {
            max_y = y1 + 1;
        }
        if y0 == y1 {
            let x0 = p0 - y0 * w;
            let x1 = p1 - y0 * w;
            if x0 < min_x {
                min_x = x0;
            }
            if x1 > max_x {
                max_x = x1;
            }
        } else {
            let x0 = p0 - y0 * w;
            let x1 = p1 - y1 * w;
            if y1 > y0 + 1 {
                min_x = 0;
                max_x = w;
            } else {
                if x0 < min_x {
                    min_x = x0;
                }
                max_x = w;
                if x1 > 0 {
                    min_x = 0;
                }
                if x1 > max_x {
                    max_x = x1;
                }
            }
        }

        let mut r = eq;
        while r >= 255 {
            runs.push(255);
            r -= 255;
        }
        runs.push(r as u8);

        let mut d = diffs;
        while d >= 255 {
            runs.push(255);
            d -= 255;
        }
        runs.push(d as u8);

        let mut k = start;
        while k < i {
            dr.push(cur[k].wrapping_sub(prev[k]));
            dg.push(cur[k + 1].wrapping_sub(prev[k + 1]));
            db.push(cur[k + 2].wrapping_sub(prev[k + 2]));
            k += 3;
        }

        // Skip the 1 known-equal pixel that ended `count_diff_rgb`.
        i += 3;
    }

    if runs.is_empty() {
        return None;
    }

    let runs_len = runs.len() as u32;
    let n_px = db.len() as u32;
    out.reserve(8 + runs.len() + db.len() * 3);
    out.extend_from_slice(&runs_len.to_le_bytes());
    out.extend_from_slice(&n_px.to_le_bytes());
    out.extend_from_slice(&runs);
    out.extend_from_slice(&db);
    out.extend_from_slice(&dg);
    out.extend_from_slice(&dr);

    Some(PackedMeta {
        x0: min_x as u16,
        y0: min_y as u16,
        x1: max_x as u16,
        y1: max_y as u16,
        tile_mask,
    })
}

/// Apply a planar-delta BitPack frame directly onto a BGRX canvas (`w * h * 4`)
/// in place with full bounds validation (used on loop 0).
pub fn unpack_planar_delta_bgrx(canvas_bgrx: &mut [u8], blob: &[u8]) -> Result<(), &'static str> {
    if blob.len() < 8 {
        return Err("planar delta header too short");
    }
    let runs_len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
    let n_px = u32::from_le_bytes(blob[4..8].try_into().unwrap()) as usize;
    let expected = 8usize
        .checked_add(runs_len)
        .and_then(|v| v.checked_add(n_px * 3))
        .ok_or("planar delta length overflow")?;
    if blob.len() != expected {
        return Err("planar delta size mismatch");
    }

    let runs = &blob[8..8 + runs_len];
    let db = &blob[8 + runs_len..8 + runs_len + n_px];
    let dg = &blob[8 + runs_len + n_px..8 + runs_len + n_px * 2];
    let dr = &blob[8 + runs_len + n_px * 2..];

    let max_px = canvas_bgrx.len() / 4;
    let mut rp = 0usize;
    let mut dp = 0usize; // pixel index in canvas
    let mut cp = 0usize; // pixel index in planar channels

    while rp < runs_len {
        let mut skip = 0usize;
        while rp < runs_len && runs[rp] == 255 {
            skip += 255;
            rp += 1;
        }
        if rp >= runs_len {
            return Err("truncated skip");
        }
        skip += runs[rp] as usize;
        rp += 1;
        dp += skip;

        let mut cpy = 0usize;
        while rp < runs_len && runs[rp] == 255 {
            cpy += 255;
            rp += 1;
        }
        if rp >= runs_len {
            return Err("truncated copy len");
        }
        cpy += runs[rp] as usize;
        rp += 1;

        if dp + cpy > max_px || cp + cpy > n_px {
            return Err("planar delta out of bounds");
        }

        let dst = canvas_bgrx.as_mut_ptr();
        unsafe {
            for k in 0..cpy {
                let p = dst.add((dp + k) * 4);
                let ci = cp + k;
                *p = (*p).wrapping_add(*db.get_unchecked(ci));
                *p.add(1) = (*p.add(1)).wrapping_add(*dg.get_unchecked(ci));
                *p.add(2) = (*p.add(2)).wrapping_add(*dr.get_unchecked(ci));
            }
        }
        cp += cpy;
        dp += cpy + 1;
    }

    if cp != n_px {
        return Err("planar delta pixel count mismatch");
    }
    Ok(())
}

/// Fast unchecked planar-delta BGRX unpacker for frames already validated on loop 0.
#[inline]
pub unsafe fn unpack_planar_delta_bgrx_unchecked(canvas_bgrx: &mut [u8], blob: &[u8]) {
    let runs_len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
    let n_px = u32::from_le_bytes(blob[4..8].try_into().unwrap()) as usize;

    let mut r_ptr = blob.as_ptr().add(8);
    let r_end = r_ptr.add(runs_len);
    let mut b_ptr = r_end;
    let mut g_ptr = b_ptr.add(n_px);
    let mut rd_ptr = g_ptr.add(n_px);
    let mut dst = canvas_bgrx.as_mut_ptr();

    while r_ptr < r_end {
        while *r_ptr == 255 {
            dst = dst.add(255 * 4);
            r_ptr = r_ptr.add(1);
        }
        dst = dst.add((*r_ptr as usize) * 4);
        r_ptr = r_ptr.add(1);

        let mut cpy = 0usize;
        while *r_ptr == 255 {
            cpy += 255;
            r_ptr = r_ptr.add(1);
        }
        cpy += *r_ptr as usize;
        r_ptr = r_ptr.add(1);

        for _ in 0..cpy {
            *dst = (*dst).wrapping_add(*b_ptr);
            *dst.add(1) = (*dst.add(1)).wrapping_add(*g_ptr);
            *dst.add(2) = (*dst.add(2)).wrapping_add(*rd_ptr);
            dst = dst.add(4);
            b_ptr = b_ptr.add(1);
            g_ptr = g_ptr.add(1);
            rd_ptr = rd_ptr.add(1);
        }
        dst = dst.add(4);
    }
}

// ============================================================================
// `.mwa` v3 binary cache writer & zero-PSS `pread` streaming reader
// ============================================================================

/// Atomically write a complete `.mwa` v3 cache file and prune older cache files.
pub fn write_mwa_atomic(
    out_path: &Path,
    cell_w: u32,
    cell_h: u32,
    fps: u32,
    src_hash: u64,
    base_rgb: &[u8],
    deltas: &[(Vec<u8>, u32, u16, u16, u16, u16, [u16; TILE_ROWS])],
) -> Result<(), String> {
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir failed: {e}"))?;
    }

    let base_planar = encode_base_planar(base_rgb, cell_w as usize, cell_h as usize);
    let base_comp = lz4_compress(&base_planar);
    let frame_count = deltas.len() as u32;
    let table_offset = HEADER_SIZE as u64;
    let base_offset = table_offset + (deltas.len() * ENTRY_SIZE) as u64;
    let mut cur_off = base_offset + base_comp.len() as u64;

    let mut entries = Vec::with_capacity(deltas.len());
    for (blob, raw_len, x0, y0, x1, y1, tmask) in deltas {
        if blob.is_empty() {
            entries.push(FrameEntry::empty());
        } else {
            entries.push(FrameEntry {
                data_offset: cur_off,
                comp_len: blob.len() as u32,
                raw_len: *raw_len,
                dirty_x0: *x0,
                dirty_y0: *y0,
                dirty_x1: *x1,
                dirty_y1: *y1,
                tile_mask: *tmask,
            });
            cur_off += blob.len() as u64;
        }
    }

    let mut header = [0u8; HEADER_SIZE];
    header[0..8].copy_from_slice(&MWA_MAGIC);
    header[8..12].copy_from_slice(&MWA_VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&0u32.to_le_bytes());
    header[16..20].copy_from_slice(&cell_w.to_le_bytes());
    header[20..24].copy_from_slice(&cell_h.to_le_bytes());
    header[24..28].copy_from_slice(&frame_count.to_le_bytes());
    header[28..32].copy_from_slice(&fps.to_le_bytes());
    header[32..40].copy_from_slice(&src_hash.to_le_bytes());
    header[40..48].copy_from_slice(&base_offset.to_le_bytes());
    header[48..52].copy_from_slice(&(base_comp.len() as u32).to_le_bytes());
    header[52..60].copy_from_slice(&table_offset.to_le_bytes());

    let tmp_path = out_path.with_extension(format!("mwa.{}.tmp", unsafe { crate::getpid() }));
    let mut f = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&tmp_path)
        .map_err(|e| format!("open tmp failed: {e}"))?;

    let mut write_all = || -> std::io::Result<()> {
        f.write_all(&header)?;
        for e in &entries {
            let mut buf = [0u8; ENTRY_SIZE];
            buf[0..8].copy_from_slice(&e.data_offset.to_le_bytes());
            buf[8..12].copy_from_slice(&e.comp_len.to_le_bytes());
            buf[12..16].copy_from_slice(&e.raw_len.to_le_bytes());
            buf[16..18].copy_from_slice(&e.dirty_x0.to_le_bytes());
            buf[18..20].copy_from_slice(&e.dirty_y0.to_le_bytes());
            buf[20..22].copy_from_slice(&e.dirty_x1.to_le_bytes());
            buf[22..24].copy_from_slice(&e.dirty_y1.to_le_bytes());
            for (ti, &tm) in e.tile_mask.iter().enumerate() {
                let o = 24 + ti * 2;
                buf[o..o + 2].copy_from_slice(&tm.to_le_bytes());
            }
            f.write_all(&buf)?;
        }
        f.write_all(&base_comp)?;
        for (blob, _, _, _, _, _, _) in deltas {
            if !blob.is_empty() {
                f.write_all(blob)?;
            }
        }
        f.flush()?;
        Ok(())
    };

    if let Err(e) = write_all() {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!("write mwa failed: {e}"));
    }
    drop(f);
    fs::rename(&tmp_path, out_path).map_err(|e| {
        let _ = fs::remove_file(&tmp_path);
        format!("rename mwa failed: {e}")
    })?;

    prune_cache_dir(CACHE_KEEP_NEWEST, Some(out_path));
    Ok(())
}

/// Zero-PSS `pread` streaming reader for a `.mwa` v3 animation cache.
/// Never maps the multi-MB animation file into process address space: reads
/// each compressed delta frame on demand from OS page cache into a tiny
/// reusable buffer (~100 KB), slashing process PSS by ~7-12 MiB!
pub struct MwaFile {
    file: File,
    len: usize,
    pub cell_w: u32,
    pub cell_h: u32,
    pub frame_count: u32,
    pub fps: u32,
    pub src_hash: u64,
    pub base_offset: u64,
    pub base_comp_len: u32,
    pub entries: Vec<FrameEntry>,
    pub max_comp_delta: usize,
    pub max_raw_delta: usize,
}

impl MwaFile {
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let len = file
            .metadata()
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len() as usize;
        if len < HEADER_SIZE {
            return Err("mwa file too small".into());
        }

        let mut header = [0u8; HEADER_SIZE];
        file.read_exact_at(&mut header, 0)
            .map_err(|e| format!("read header: {e}"))?;

        if header[0..8] != MWA_MAGIC {
            return Err("invalid MWA3 magic".into());
        }
        let ver = u32::from_le_bytes(header[8..12].try_into().unwrap());
        if ver != MWA_VERSION {
            return Err(format!("unsupported MWA version {ver}"));
        }
        let cell_w = u32::from_le_bytes(header[16..20].try_into().unwrap());
        let cell_h = u32::from_le_bytes(header[20..24].try_into().unwrap());
        let frame_count = u32::from_le_bytes(header[24..28].try_into().unwrap());
        let fps = u32::from_le_bytes(header[28..32].try_into().unwrap()).clamp(1, 60);
        let src_hash = u64::from_le_bytes(header[32..40].try_into().unwrap());
        let base_offset = u64::from_le_bytes(header[40..48].try_into().unwrap());
        let base_comp_len = u32::from_le_bytes(header[48..52].try_into().unwrap());
        let table_offset = u64::from_le_bytes(header[52..60].try_into().unwrap()) as usize;

        if cell_w == 0 || cell_h == 0 || frame_count == 0 || cell_w > 1920 || cell_h > 1080 {
            return Err("invalid MWA dimensions".into());
        }
        let table_bytes = (frame_count as usize) * ENTRY_SIZE;
        let table_end = table_offset + table_bytes;
        if table_end > len || (base_offset as usize) + (base_comp_len as usize) > len {
            return Err("truncated MWA header/table".into());
        }

        let mut raw_table = vec![0u8; table_bytes];
        file.read_exact_at(&mut raw_table, table_offset as u64)
            .map_err(|e| format!("read table: {e}"))?;

        let mut entries = Vec::with_capacity(frame_count as usize);
        let mut max_comp_delta = 0usize;
        let mut max_raw_delta = 0usize;
        for i in 0..(frame_count as usize) {
            let off = i * ENTRY_SIZE;
            let data_offset = u64::from_le_bytes(raw_table[off..off + 8].try_into().unwrap());
            let comp_len = u32::from_le_bytes(raw_table[off + 8..off + 12].try_into().unwrap());
            let raw_len = u32::from_le_bytes(raw_table[off + 12..off + 16].try_into().unwrap());
            let dirty_x0 = u16::from_le_bytes(raw_table[off + 16..off + 18].try_into().unwrap());
            let dirty_y0 = u16::from_le_bytes(raw_table[off + 18..off + 20].try_into().unwrap());
            let dirty_x1 = u16::from_le_bytes(raw_table[off + 20..off + 22].try_into().unwrap());
            let dirty_y1 = u16::from_le_bytes(raw_table[off + 22..off + 24].try_into().unwrap());
            let mut tile_mask = [0u16; TILE_ROWS];
            for ti in 0..TILE_ROWS {
                let to = off + 24 + ti * 2;
                tile_mask[ti] = u16::from_le_bytes(raw_table[to..to + 2].try_into().unwrap());
            }
            if comp_len > 0 && (data_offset as usize) + (comp_len as usize) > len {
                return Err("truncated MWA delta blob".into());
            }
            if (comp_len as usize) > max_comp_delta {
                max_comp_delta = comp_len as usize;
            }
            if (raw_len as usize) > max_raw_delta {
                max_raw_delta = raw_len as usize;
            }
            entries.push(FrameEntry {
                data_offset,
                comp_len,
                raw_len,
                dirty_x0,
                dirty_y0,
                dirty_x1,
                dirty_y1,
                tile_mask,
            });
        }

        unsafe {
            let _ = posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_SEQUENTIAL);
        }

        Ok(Self {
            file,
            len,
            cell_w,
            cell_h,
            frame_count,
            fps,
            src_hash,
            base_offset,
            base_comp_len,
            entries,
            max_comp_delta,
            max_raw_delta,
        })
    }

    pub fn decode_base_bgrx(&self, dst_bgrx: &mut [u8]) -> Result<(), &'static str> {
        let comp_len = self.base_comp_len as usize;
        let mut comp = vec![0u8; comp_len];
        self.file
            .read_exact_at(&mut comp, self.base_offset)
            .map_err(|_| "pread base frame failed")?;
        let raw_len = (self.cell_w as usize) * (self.cell_h as usize) * 3;
        let mut planar = vec![0u8; raw_len];
        let got = lz4_decompress(&comp, &mut planar)?;
        if got != raw_len {
            return Err("base frame size mismatch");
        }
        decode_base_planar_bgrx(&planar, self.cell_w as usize, self.cell_h as usize, dst_bgrx)
    }

    #[inline]
    pub fn read_delta_into(&self, frame_idx: usize, dst: &mut [u8]) -> Result<usize, &'static str> {
        let e = &self.entries[frame_idx];
        let cl = e.comp_len as usize;
        if cl == 0 {
            return Ok(0);
        }
        if dst.len() < cl {
            return Err("comp scratch too small");
        }
        self.file
            .read_exact_at(&mut dst[..cl], e.data_offset)
            .map_err(|_| "pread delta failed")?;
        Ok(cl)
    }

    pub fn advise_dontneed(&self) {
        unsafe {
            let _ = posix_fadvise(self.file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED);
        }
    }

    pub fn advise_sequential(&self) {
        unsafe {
            let _ = posix_fadvise(self.file.as_raw_fd(), 0, 0, POSIX_FADV_SEQUENTIAL);
        }
    }

    pub fn file_size(&self) -> usize {
        self.len
    }
}

/// Fast built-in codec integrity test (`misa-wall --selftest`).
pub fn run_selftest() -> Result<(), String> {
    let w = 64usize;
    let h = 36usize;
    let mut f0 = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            f0[idx] = (x * 4) as u8;
            f0[idx + 1] = (y * 7) as u8;
            f0[idx + 2] = ((x ^ y) * 3) as u8;
        }
    }
    let mut f1 = f0.clone();
    for y in 10..22 {
        for x in 15..33 {
            let idx = (y * w + x) * 3;
            f1[idx] = f1[idx].wrapping_add(37);
            f1[idx + 1] = f1[idx + 1].wrapping_add(19);
            f1[idx + 2] = 200;
        }
    }
    // Single isolated pixel at (58, 31)
    let p = (31 * w + 58) * 3;
    f1[p] = 123;
    f1[p + 1] = 45;
    f1[p + 2] = 67;

    // 1. Base planar round-trip -> BGRX
    let base_p = encode_base_planar(&f0, w, h);
    let base_c = lz4_compress(&base_p);
    let mut base_d = vec![0u8; base_p.len()];
    if lz4_decompress(&base_c, &mut base_d).map_err(|e| e.to_string())? != base_p.len() {
        return Err("base LZ4 size mismatch".into());
    }
    let mut bgrx0 = vec![0u8; w * h * 4];
    decode_base_planar_bgrx(&base_d, w, h, &mut bgrx0).map_err(|e| e.to_string())?;
    for i in 0..(w * h) {
        if bgrx0[i * 4] != f0[i * 3 + 2]
            || bgrx0[i * 4 + 1] != f0[i * 3 + 1]
            || bgrx0[i * 4 + 2] != f0[i * 3]
        {
            return Err("base planar BGRX mismatch".into());
        }
    }

    // 2. Planar-delta + tile mask + LZ4 round-trip
    let mut raw_delta = Vec::new();
    let meta = pack_planar_delta(&f0, &f1, w, h, &mut raw_delta)
        .ok_or_else(|| "expected non-empty delta".to_string())?;
    if (meta.x0, meta.y0, meta.x1, meta.y1) != (15, 10, 59, 32) {
        return Err(format!(
            "dirty bbox mismatch: got {:?}, expected (15, 10, 59, 32)",
            (meta.x0, meta.y0, meta.x1, meta.y1)
        ));
    }
    let mut tile_rects = Vec::new();
    merge_tile_rects(
        &meta.tile_mask,
        (meta.x0, meta.y0, meta.x1, meta.y1),
        w as i32,
        h as i32,
        &mut tile_rects,
    );
    if tile_rects.len() < 2 {
        return Err(format!("expected >=2 disjoint dirty tile rects, got {tile_rects:?}"));
    }

    let comp = lz4_compress(&raw_delta);
    let mut decomp = vec![0u8; raw_delta.len()];
    let got = lz4_decompress(&comp, &mut decomp).map_err(|e| e.to_string())?;
    if got != raw_delta.len() || decomp != raw_delta {
        return Err("LZ4 round-trip mismatch".into());
    }

    let mut bgrx1 = bgrx0.clone();
    unpack_planar_delta_bgrx(&mut bgrx1, &decomp).map_err(|e| e.to_string())?;
    let mut bgrx2 = bgrx0.clone();
    unsafe { unpack_planar_delta_bgrx_unchecked(&mut bgrx2, &decomp); }
    for i in 0..(w * h) {
        let exp = [f1[i * 3 + 2], f1[i * 3 + 1], f1[i * 3], 0];
        if bgrx1[i * 4..i * 4 + 4] != exp || bgrx2[i * 4..i * 4 + 4] != exp {
            return Err(format!("planar delta unpack mismatch at pixel {i}"));
        }
    }
    Ok(())
}
