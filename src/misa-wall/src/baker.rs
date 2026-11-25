//! One-shot offscreen OpenGL shader baker for `misa-wall` v2.
//!
//! `libGL.so.1` is loaded dynamically via `dlopen` ONLY inside this module
//! during transient `--bake-frame` (CardPicker Theme Showcase) or `--bake-loop`
//! (one-time `.mwa` cache generation) calls, which exit immediately afterward so
//! the persistent desktop wallpaper daemon uses 0 MiB of Mesa/GLX RAM forever!

use std::ffi::{c_char, c_int, c_uchar, c_uint, c_ulong, c_void, CStr, CString};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::ptr::{null, null_mut};

use crate::codec::{lz4_compress, pack_planar_delta, write_mwa_atomic, TILE_ROWS};
use crate::{
    decode_rgb_image_scaled, getpid, ignore_x_err, Display, Visual, Window, XDefaultRootWindow,
    XDefaultScreen, XFree, XOpenDisplay, XSetErrorHandler, XSync,
};

const RTLD_LAZY: c_int = 1;
const RTLD_LOCAL: c_int = 0;

const INPUT_OUTPUT: c_uint = 1;
const ALLOC_NONE: c_int = 0;
const CW_BACK_PIXMAP: c_ulong = 1 << 0;
const CW_BORDER_PIXEL: c_ulong = 1 << 3;
const CW_OVERRIDE_REDIRECT: c_ulong = 1 << 9;
const CW_COLORMAP: c_ulong = 1 << 13;

const GLX_RGBA: c_int = 4;
const GLX_DOUBLEBUFFER: c_int = 5;
const GLX_RED_SIZE: c_int = 8;
const GLX_GREEN_SIZE: c_int = 9;
const GLX_BLUE_SIZE: c_int = 10;

const GL_TRIANGLES: u32 = 0x0004;
const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_FLOAT: u32 = 0x1406;
const GL_RGB: u32 = 0x1907;
const GL_RGBA: u32 = 0x1908;
const GL_NEAREST: i32 = 0x2600;
const GL_LINEAR: i32 = 0x2601;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_WRAP_S: u32 = 0x2802;
const GL_TEXTURE_WRAP_T: u32 = 0x2803;
const GL_CLAMP_TO_EDGE: i32 = 0x812F;
const GL_UNPACK_ALIGNMENT: u32 = 0x0CF5;
const GL_PACK_ALIGNMENT: u32 = 0x0D05;
const GL_TEXTURE0: u32 = 0x84C0;
const GL_ARRAY_BUFFER: u32 = 0x8892;
const GL_STATIC_DRAW: u32 = 0x88E4;
const GL_FRAGMENT_SHADER: u32 = 0x8B30;
const GL_VERTEX_SHADER: u32 = 0x8B31;
const GL_COMPILE_STATUS: u32 = 0x8B81;
const GL_LINK_STATUS: u32 = 0x8B82;
const GL_FRAMEBUFFER: u32 = 0x8D40;
const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;

type Colormap = c_ulong;
type GLXContext = *mut c_void;

#[repr(C)]
struct XVisualInfo {
    visual: *mut Visual,
    visualid: c_ulong,
    screen: c_int,
    depth: c_int,
    class: c_int,
    red_mask: c_ulong,
    green_mask: c_ulong,
    blue_mask: c_ulong,
    colormap_size: c_int,
    bits_per_rgb: c_int,
}

#[repr(C)]
struct XSetWindowAttributes {
    background_pixmap: c_ulong,
    background_pixel: c_ulong,
    border_pixmap: c_ulong,
    border_pixel: c_ulong,
    bit_gravity: c_int,
    win_gravity: c_int,
    backing_store: c_int,
    backing_planes: c_ulong,
    backing_pixel: c_ulong,
    save_under: c_int,
    event_mask: i64,
    do_not_propagate_mask: i64,
    override_redirect: c_int,
    colormap: Colormap,
    cursor: c_ulong,
}

#[link(name = "X11")]
extern "C" {
    fn XCreateColormap(d: *mut Display, w: Window, v: *mut Visual, alloc: c_int) -> Colormap;
    fn XFreeColormap(d: *mut Display, c: Colormap) -> c_int;
    fn XCreateWindow(
        d: *mut Display,
        parent: Window,
        x: c_int,
        y: c_int,
        w: c_uint,
        h: c_uint,
        bw: c_uint,
        depth: c_int,
        class: c_uint,
        vis: *mut Visual,
        mask: c_ulong,
        attrs: *mut XSetWindowAttributes,
    ) -> Window;
    fn XDestroyWindow(d: *mut Display, w: Window) -> c_int;
    fn XCloseDisplay(d: *mut Display) -> c_int;
}

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

type FnGlXChooseVisual = unsafe extern "C" fn(*mut Display, c_int, *mut c_int) -> *mut XVisualInfo;
type FnGlXCreateContext =
    unsafe extern "C" fn(*mut Display, *mut XVisualInfo, GLXContext, c_int) -> GLXContext;
type FnGlXDestroyContext = unsafe extern "C" fn(*mut Display, GLXContext);
type FnGlXMakeCurrent = unsafe extern "C" fn(*mut Display, Window, GLXContext) -> c_int;
type FnGlXGetProcAddress = unsafe extern "C" fn(*const c_uchar) -> *mut c_void;

type FnGlViewport = unsafe extern "C" fn(i32, i32, i32, i32);
type FnGlCreateShader = unsafe extern "C" fn(u32) -> u32;
type FnGlShaderSource = unsafe extern "C" fn(u32, i32, *const *const c_char, *const i32);
type FnGlCompileShader = unsafe extern "C" fn(u32);
type FnGlGetShaderiv = unsafe extern "C" fn(u32, u32, *mut i32);
type FnGlGetShaderInfoLog = unsafe extern "C" fn(u32, i32, *mut i32, *mut c_char);
type FnGlDeleteShader = unsafe extern "C" fn(u32);
type FnGlCreateProgram = unsafe extern "C" fn() -> u32;
type FnGlAttachShader = unsafe extern "C" fn(u32, u32);
type FnGlBindAttribLocation = unsafe extern "C" fn(u32, u32, *const c_char);
type FnGlLinkProgram = unsafe extern "C" fn(u32);
type FnGlGetProgramiv = unsafe extern "C" fn(u32, u32, *mut i32);
type FnGlDeleteProgram = unsafe extern "C" fn(u32);
type FnGlUseProgram = unsafe extern "C" fn(u32);
type FnGlGetUniformLocation = unsafe extern "C" fn(u32, *const c_char) -> i32;
type FnGlUniform1i = unsafe extern "C" fn(i32, i32);
type FnGlUniform1f = unsafe extern "C" fn(i32, f32);
type FnGlUniform2f = unsafe extern "C" fn(i32, f32, f32);
type FnGlUniform4f = unsafe extern "C" fn(i32, f32, f32, f32, f32);
type FnGlGenBuffers = unsafe extern "C" fn(i32, *mut u32);
type FnGlDeleteBuffers = unsafe extern "C" fn(i32, *const u32);
type FnGlBindBuffer = unsafe extern "C" fn(u32, u32);
type FnGlBufferData = unsafe extern "C" fn(u32, isize, *const c_void, u32);
type FnGlEnableVertexAttribArray = unsafe extern "C" fn(u32);
type FnGlVertexAttribPointer = unsafe extern "C" fn(u32, i32, u32, u8, i32, *const c_void);
type FnGlGenTextures = unsafe extern "C" fn(i32, *mut u32);
type FnGlDeleteTextures = unsafe extern "C" fn(i32, *const u32);
type FnGlActiveTexture = unsafe extern "C" fn(u32);
type FnGlBindTexture = unsafe extern "C" fn(u32, u32);
type FnGlTexParameteri = unsafe extern "C" fn(u32, u32, i32);
type FnGlPixelStorei = unsafe extern "C" fn(u32, i32);
type FnGlTexImage2D = unsafe extern "C" fn(u32, i32, i32, i32, i32, i32, u32, u32, *const c_void);
type FnGlDrawArrays = unsafe extern "C" fn(u32, i32, i32);
type FnGlReadPixels = unsafe extern "C" fn(i32, i32, i32, i32, u32, u32, *mut c_void);
type FnGlGenFramebuffers = unsafe extern "C" fn(i32, *mut u32);
type FnGlDeleteFramebuffers = unsafe extern "C" fn(i32, *const u32);
type FnGlBindFramebuffer = unsafe extern "C" fn(u32, u32);
type FnGlFramebufferTexture2D = unsafe extern "C" fn(u32, u32, u32, u32, i32);
type FnGlCheckFramebufferStatus = unsafe extern "C" fn(u32) -> u32;

struct BakerGl {
    choose_visual: FnGlXChooseVisual,
    create_context: FnGlXCreateContext,
    destroy_context: FnGlXDestroyContext,
    make_current: FnGlXMakeCurrent,
    viewport: FnGlViewport,
    create_shader: FnGlCreateShader,
    shader_source: FnGlShaderSource,
    compile_shader: FnGlCompileShader,
    get_shaderiv: FnGlGetShaderiv,
    get_shader_info_log: FnGlGetShaderInfoLog,
    delete_shader: Option<FnGlDeleteShader>,
    create_program: FnGlCreateProgram,
    attach_shader: FnGlAttachShader,
    bind_attrib_location: FnGlBindAttribLocation,
    link_program: FnGlLinkProgram,
    get_programiv: FnGlGetProgramiv,
    delete_program: Option<FnGlDeleteProgram>,
    use_program: FnGlUseProgram,
    get_uniform_location: FnGlGetUniformLocation,
    uniform1i: FnGlUniform1i,
    uniform1f: FnGlUniform1f,
    uniform2f: FnGlUniform2f,
    uniform4f: FnGlUniform4f,
    gen_buffers: FnGlGenBuffers,
    delete_buffers: Option<FnGlDeleteBuffers>,
    bind_buffer: FnGlBindBuffer,
    buffer_data: FnGlBufferData,
    enable_vertex_attrib_array: FnGlEnableVertexAttribArray,
    vertex_attrib_pointer: FnGlVertexAttribPointer,
    gen_textures: FnGlGenTextures,
    delete_textures: FnGlDeleteTextures,
    active_texture: FnGlActiveTexture,
    bind_texture: FnGlBindTexture,
    tex_parameteri: FnGlTexParameteri,
    pixel_storei: FnGlPixelStorei,
    tex_image_2d: FnGlTexImage2D,
    draw_arrays: FnGlDrawArrays,
    read_pixels: FnGlReadPixels,
    gen_framebuffers: FnGlGenFramebuffers,
    delete_framebuffers: FnGlDeleteFramebuffers,
    bind_framebuffer: FnGlBindFramebuffer,
    framebuffer_texture_2d: FnGlFramebufferTexture2D,
    check_framebuffer_status: FnGlCheckFramebufferStatus,
}

const LIVE_VS_SRC: &str = "#version 120
attribute vec2 aPos;
varying vec2 vUv;
void main() {
    vUv = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
    gl_Position = vec4(aPos, 0.0, 1.0);
}
";

unsafe fn load_sym(lib: *mut c_void, gpa: Option<FnGlXGetProcAddress>, name: &str) -> *mut c_void {
    let cs = CString::new(name).unwrap();
    if let Some(f) = gpa {
        let p = f(cs.as_ptr() as *const c_uchar);
        if !p.is_null() {
            return p;
        }
    }
    dlsym(lib, cs.as_ptr())
}

unsafe fn load_gl() -> Option<BakerGl> {
    let mut lib = null_mut();
    for soname in ["libGL.so.1", "libGL.so"] {
        let cs = CString::new(soname).unwrap();
        lib = dlopen(cs.as_ptr(), RTLD_LAZY | RTLD_LOCAL);
        if !lib.is_null() {
            break;
        }
    }
    if lib.is_null() {
        return None;
    }

    let gpa_ptr = {
        let a = dlsym(lib, CString::new("glXGetProcAddressARB").unwrap().as_ptr());
        if !a.is_null() {
            a
        } else {
            dlsym(lib, CString::new("glXGetProcAddress").unwrap().as_ptr())
        }
    };
    let gpa: Option<FnGlXGetProcAddress> = if gpa_ptr.is_null() {
        None
    } else {
        Some(std::mem::transmute(gpa_ptr))
    };

    macro_rules! req {
        ($n:expr, $t:ty) => {{
            let p = load_sym(lib, gpa, $n);
            if p.is_null() {
                return None;
            }
            std::mem::transmute::<*mut c_void, $t>(p)
        }};
    }
    macro_rules! opt {
        ($n:expr, $t:ty) => {{
            let p = load_sym(lib, gpa, $n);
            if p.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*mut c_void, $t>(p))
            }
        }};
    }

    Some(BakerGl {
        choose_visual: req!("glXChooseVisual", FnGlXChooseVisual),
        create_context: req!("glXCreateContext", FnGlXCreateContext),
        destroy_context: req!("glXDestroyContext", FnGlXDestroyContext),
        make_current: req!("glXMakeCurrent", FnGlXMakeCurrent),
        viewport: req!("glViewport", FnGlViewport),
        create_shader: req!("glCreateShader", FnGlCreateShader),
        shader_source: req!("glShaderSource", FnGlShaderSource),
        compile_shader: req!("glCompileShader", FnGlCompileShader),
        get_shaderiv: req!("glGetShaderiv", FnGlGetShaderiv),
        get_shader_info_log: req!("glGetShaderInfoLog", FnGlGetShaderInfoLog),
        delete_shader: opt!("glDeleteShader", FnGlDeleteShader),
        create_program: req!("glCreateProgram", FnGlCreateProgram),
        attach_shader: req!("glAttachShader", FnGlAttachShader),
        bind_attrib_location: req!("glBindAttribLocation", FnGlBindAttribLocation),
        link_program: req!("glLinkProgram", FnGlLinkProgram),
        get_programiv: req!("glGetProgramiv", FnGlGetProgramiv),
        delete_program: opt!("glDeleteProgram", FnGlDeleteProgram),
        use_program: req!("glUseProgram", FnGlUseProgram),
        get_uniform_location: req!("glGetUniformLocation", FnGlGetUniformLocation),
        uniform1i: req!("glUniform1i", FnGlUniform1i),
        uniform1f: req!("glUniform1f", FnGlUniform1f),
        uniform2f: req!("glUniform2f", FnGlUniform2f),
        uniform4f: req!("glUniform4f", FnGlUniform4f),
        gen_buffers: req!("glGenBuffers", FnGlGenBuffers),
        delete_buffers: opt!("glDeleteBuffers", FnGlDeleteBuffers),
        bind_buffer: req!("glBindBuffer", FnGlBindBuffer),
        buffer_data: req!("glBufferData", FnGlBufferData),
        enable_vertex_attrib_array: req!("glEnableVertexAttribArray", FnGlEnableVertexAttribArray),
        vertex_attrib_pointer: req!("glVertexAttribPointer", FnGlVertexAttribPointer),
        gen_textures: req!("glGenTextures", FnGlGenTextures),
        delete_textures: req!("glDeleteTextures", FnGlDeleteTextures),
        active_texture: req!("glActiveTexture", FnGlActiveTexture),
        bind_texture: req!("glBindTexture", FnGlBindTexture),
        tex_parameteri: req!("glTexParameteri", FnGlTexParameteri),
        pixel_storei: req!("glPixelStorei", FnGlPixelStorei),
        tex_image_2d: req!("glTexImage2D", FnGlTexImage2D),
        draw_arrays: req!("glDrawArrays", FnGlDrawArrays),
        read_pixels: req!("glReadPixels", FnGlReadPixels),
        gen_framebuffers: req!("glGenFramebuffers", FnGlGenFramebuffers),
        delete_framebuffers: req!("glDeleteFramebuffers", FnGlDeleteFramebuffers),
        bind_framebuffer: req!("glBindFramebuffer", FnGlBindFramebuffer),
        framebuffer_texture_2d: req!("glFramebufferTexture2D", FnGlFramebufferTexture2D),
        check_framebuffer_status: req!("glCheckFramebufferStatus", FnGlCheckFramebufferStatus),
    })
}

unsafe fn compile_shader(gl: &BakerGl, kind: u32, src: &str) -> Option<u32> {
    let sh = (gl.create_shader)(kind);
    if sh == 0 {
        return None;
    }
    let cs = CString::new(src).ok()?;
    let ptr = cs.as_ptr();
    (gl.shader_source)(sh, 1, &ptr, null());
    (gl.compile_shader)(sh);
    let mut ok: i32 = 0;
    (gl.get_shaderiv)(sh, GL_COMPILE_STATUS, &mut ok);
    if ok == 0 {
        let mut buf = [0i8; 1024];
        let mut len: i32 = 0;
        (gl.get_shader_info_log)(sh, buf.len() as i32, &mut len, buf.as_mut_ptr());
        let msg = CStr::from_ptr(buf.as_ptr()).to_string_lossy();
        eprintln!("misa-wall baker: shader compile error:\n{msg}");
        return None;
    }
    Some(sh)
}

unsafe fn link_prog(gl: &BakerGl, fs_src: &str) -> Option<u32> {
    let vs = compile_shader(gl, GL_VERTEX_SHADER, LIVE_VS_SRC)?;
    let Some(fs_sh) = compile_shader(gl, GL_FRAGMENT_SHADER, fs_src) else {
        if let Some(del) = gl.delete_shader {
            del(vs);
        }
        return None;
    };
    let prog = (gl.create_program)();
    (gl.attach_shader)(prog, vs);
    (gl.attach_shader)(prog, fs_sh);
    let c_apos = CString::new("aPos").unwrap();
    (gl.bind_attrib_location)(prog, 0, c_apos.as_ptr());
    (gl.link_program)(prog);
    if let Some(del) = gl.delete_shader {
        del(vs);
        del(fs_sh);
    }
    let mut linked: i32 = 0;
    (gl.get_programiv)(prog, GL_LINK_STATUS, &mut linked);
    if linked == 0 {
        if let Some(delp) = gl.delete_program {
            delp(prog);
        }
        return None;
    }
    Some(prog)
}

struct OffscreenRenderer {
    dpy: *mut Display,
    win: Window,
    cmap: Colormap,
    ctx: GLXContext,
    gl: BakerGl,
    prog: u32,
    fbo: u32,
    fbo_tex: u32,
    img_tex: u32,
    vbo: u32,
    loc_time: i32,
    fw: i32,
    fh: i32,
}

impl OffscreenRenderer {
    unsafe fn new(img_path: &str, frag_path: &str, fw: i32, fh: i32) -> Option<Self> {
        let frag_src = fs::read_to_string(frag_path).ok()?;
        let gl = load_gl()?;
        let dpy = XOpenDisplay(null());
        if dpy.is_null() {
            return None;
        }
        XSetErrorHandler(Some(ignore_x_err));
        let scr = XDefaultScreen(dpy);
        let root = XDefaultRootWindow(dpy);

        let (tex_w, tex_h, tex_bytes) = decode_rgb_image_scaled(img_path, fw * 2, fh * 2)?;
        let im_aspect = (tex_w as f32) / (tex_h.max(1) as f32);
        let vp_a = (fw as f32) / (fh.max(1) as f32);
        let cover = if vp_a > im_aspect {
            let sy = im_aspect / vp_a;
            [1.0f32, sy, 0.0, (1.0 - sy) * 0.5]
        } else {
            let sx = vp_a / im_aspect;
            [sx, 1.0f32, (1.0 - sx) * 0.5, 0.0]
        };

        let mut vis_attribs = [
            GLX_RGBA,
            GLX_DOUBLEBUFFER,
            GLX_RED_SIZE,
            8,
            GLX_GREEN_SIZE,
            8,
            GLX_BLUE_SIZE,
            8,
            0,
        ];
        let vi = (gl.choose_visual)(dpy, scr, vis_attribs.as_mut_ptr());
        if vi.is_null() {
            XCloseDisplay(dpy);
            return None;
        }
        let cmap = XCreateColormap(dpy, root, (*vi).visual, ALLOC_NONE);
        let mut swa = XSetWindowAttributes {
            background_pixmap: 0,
            background_pixel: 0,
            border_pixmap: 0,
            border_pixel: 0,
            bit_gravity: 0,
            win_gravity: 0,
            backing_store: 0,
            backing_planes: 0,
            backing_pixel: 0,
            save_under: 0,
            event_mask: 0,
            do_not_propagate_mask: 0,
            override_redirect: 1,
            colormap: cmap,
            cursor: 0,
        };
        // Unmapped 1x1 hidden window: never appears on screen!
        let win = XCreateWindow(
            dpy,
            root,
            -16,
            -16,
            1,
            1,
            0,
            (*vi).depth,
            INPUT_OUTPUT,
            (*vi).visual,
            CW_BACK_PIXMAP | CW_BORDER_PIXEL | CW_OVERRIDE_REDIRECT | CW_COLORMAP,
            &mut swa,
        );
        let ctx = (gl.create_context)(dpy, vi, null_mut(), 1);
        XFree(vi as *mut c_void);
        if ctx.is_null() || (gl.make_current)(dpy, win, ctx) == 0 {
            if !ctx.is_null() {
                (gl.destroy_context)(dpy, ctx);
            }
            XDestroyWindow(dpy, win);
            XFreeColormap(dpy, cmap);
            XCloseDisplay(dpy);
            return None;
        }

        let Some(prog) = link_prog(&gl, &frag_src) else {
            (gl.make_current)(dpy, 0, null_mut());
            (gl.destroy_context)(dpy, ctx);
            XDestroyWindow(dpy, win);
            XFreeColormap(dpy, cmap);
            XCloseDisplay(dpy);
            return None;
        };

        let mut fbo: u32 = 0;
        let mut fbo_tex: u32 = 0;
        (gl.gen_framebuffers)(1, &mut fbo);
        (gl.gen_textures)(1, &mut fbo_tex);
        (gl.bind_texture)(GL_TEXTURE_2D, fbo_tex);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        (gl.tex_image_2d)(GL_TEXTURE_2D, 0, GL_RGBA as i32, fw, fh, 0, GL_RGBA, GL_UNSIGNED_BYTE, null());
        (gl.bind_framebuffer)(GL_FRAMEBUFFER, fbo);
        (gl.framebuffer_texture_2d)(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_tex, 0);
        if (gl.check_framebuffer_status)(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE {
            (gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
            (gl.delete_framebuffers)(1, &fbo);
            (gl.delete_textures)(1, &fbo_tex);
            if let Some(delp) = gl.delete_program {
                delp(prog);
            }
            (gl.make_current)(dpy, 0, null_mut());
            (gl.destroy_context)(dpy, ctx);
            XDestroyWindow(dpy, win);
            XFreeColormap(dpy, cmap);
            XCloseDisplay(dpy);
            return None;
        }

        let verts: [f32; 6] = [-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
        let mut vbo: u32 = 0;
        (gl.gen_buffers)(1, &mut vbo);
        (gl.bind_buffer)(GL_ARRAY_BUFFER, vbo);
        (gl.buffer_data)(
            GL_ARRAY_BUFFER,
            std::mem::size_of_val(&verts) as isize,
            verts.as_ptr() as *const c_void,
            GL_STATIC_DRAW,
        );
        (gl.enable_vertex_attrib_array)(0);
        (gl.vertex_attrib_pointer)(0, 2, GL_FLOAT, 0, 0, null());

        let mut img_tex: u32 = 0;
        (gl.active_texture)(GL_TEXTURE0);
        (gl.gen_textures)(1, &mut img_tex);
        (gl.bind_texture)(GL_TEXTURE_2D, img_tex);
        (gl.pixel_storei)(GL_UNPACK_ALIGNMENT, 1);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        (gl.tex_parameteri)(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        (gl.tex_image_2d)(
            GL_TEXTURE_2D,
            0,
            GL_RGB as i32,
            tex_w,
            tex_h,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            tex_bytes.as_ptr() as *const c_void,
        );

        let uloc = |name: &str| {
            let cs = CString::new(name).unwrap();
            (gl.get_uniform_location)(prog, cs.as_ptr())
        };
        (gl.use_program)(prog);
        let lt = uloc("uTex");
        if lt >= 0 {
            (gl.uniform1i)(lt, 0);
        }
        let lr = uloc("uRes");
        if lr >= 0 {
            (gl.uniform2f)(lr, fw as f32, fh as f32);
        }
        let lc = uloc("uCover");
        if lc >= 0 {
            (gl.uniform4f)(lc, cover[0], cover[1], cover[2], cover[3]);
        }
        let lm = uloc("uMouse");
        if lm >= 0 {
            (gl.uniform2f)(lm, 0.0, 0.0);
        }
        let lce = uloc("uCell");
        if lce >= 0 {
            (gl.uniform1f)(lce, 1.0);
        }
        let loc_time = uloc("uTime");

        (gl.viewport)(0, 0, fw, fh);
        (gl.pixel_storei)(GL_PACK_ALIGNMENT, 1);

        Some(Self {
            dpy,
            win,
            cmap,
            ctx,
            gl,
            prog,
            fbo,
            fbo_tex,
            img_tex,
            vbo,
            loc_time,
            fw,
            fh,
        })
    }

    /// Render one frame at `t_sec` into top-down `dst_rgb` (`fw * fh * 3` bytes).
    unsafe fn render_rgb_topdown(&self, t_sec: f32, gl_flip_scratch: &mut [u8], dst_rgb: &mut [u8]) {
        if self.loc_time >= 0 {
            (self.gl.uniform1f)(self.loc_time, t_sec);
        }
        (self.gl.draw_arrays)(GL_TRIANGLES, 0, 3);
        (self.gl.read_pixels)(
            0,
            0,
            self.fw,
            self.fh,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            gl_flip_scratch.as_mut_ptr() as *mut c_void,
        );
        let row = (self.fw as usize) * 3;
        let h = self.fh as usize;
        for y in 0..h {
            let src_y = h - 1 - y;
            dst_rgb[y * row..(y + 1) * row]
                .copy_from_slice(&gl_flip_scratch[src_y * row..(src_y + 1) * row]);
        }
    }
}

impl Drop for OffscreenRenderer {
    fn drop(&mut self) {
        unsafe {
            (self.gl.bind_framebuffer)(GL_FRAMEBUFFER, 0);
            (self.gl.delete_framebuffers)(1, &self.fbo);
            (self.gl.delete_textures)(1, &self.fbo_tex);
            (self.gl.delete_textures)(1, &self.img_tex);
            if let Some(delb) = self.gl.delete_buffers {
                delb(1, &self.vbo);
            }
            if let Some(delp) = self.gl.delete_program {
                delp(self.prog);
            }
            (self.gl.make_current)(self.dpy, 0, null_mut());
            (self.gl.destroy_context)(self.dpy, self.ctx);
            XDestroyWindow(self.dpy, self.win);
            XFreeColormap(self.dpy, self.cmap);
            XSync(self.dpy, 0);
            XCloseDisplay(self.dpy);
        }
    }
}

/// Offscreen single-frame baker for static Theme Picker snapshots (`--bake-frame`).
pub fn bake_shader_frame(img_path: &str, frag_path: &str, out_ppm: &str, time_sec: f32) -> i32 {
    let fw = 640i32;
    let fh = 360i32;
    unsafe {
        let Some(r) = OffscreenRenderer::new(img_path, frag_path, fw, fh) else {
            return 1;
        };
        let len = (fw as usize) * (fh as usize) * 3;
        let mut scratch = vec![0u8; len];
        let mut rgb = vec![0u8; len];
        r.render_rgb_topdown(time_sec, &mut scratch, &mut rgb);

        let mut ppm = Vec::with_capacity(32 + len);
        let _ = write!(ppm, "P6\n{fw} {fh}\n255\n");
        ppm.extend_from_slice(&rgb);
        let tmp = format!("{out_ppm}.{}.tmp", getpid());
        if fs::write(&tmp, &ppm).is_ok() && fs::rename(&tmp, out_ppm).is_ok() {
            0
        } else {
            let _ = fs::remove_file(&tmp);
            1
        }
    }
}

const BAYER4_CELL: [[u8; 4]; 4] = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
];

/// Crisp per-cell Bayer-ordered temporal cross-dissolve between `tail` (fading out)
/// and `head` (fading in) at weight `w` in `(0, 1)`. Preserves 100% pure quantized
/// 3px Bayer colors with zero blurry ghosting and zero extra palette colors!
fn bayer_temporal_dissolve(head: &[u8], tail: &[u8], w: usize, h: usize, frac: f32, out: &mut [u8]) {
    let thresh = (frac * 16.0).clamp(0.0, 16.0);
    for y in 0..h {
        let brow = BAYER4_CELL[y & 3];
        let row_off = y * w * 3;
        for x in 0..w {
            let off = row_off + x * 3;
            // Hash cell slightly with its 4x4 block position so overlap dissolves organically.
            let b = brow[x & 3] as f32 + 0.5;
            let src = if b < thresh { head } else { tail };
            out[off] = src[off];
            out[off + 1] = src[off + 1];
            out[off + 2] = src[off + 2];
        }
    }
}

/// One-shot seamless animation loop baker (`--bake-loop`).
/// Renders `frames + overlap` frames at `fw x fh` on a hidden 1x1 GLX FBO,
/// dissolves the tail cleanly back into frame 0 via crisp ordered-Bayer cell
/// selection (zero loop seam/pop!), encodes every frame transition (including
/// `frames-1 -> 0`) with `awww`-style BitPack + LZ4 + exact dirty bounding box,
/// and writes the `.mwa` cache file atomically before exiting.
pub fn bake_shader_loop(
    img_path: &str,
    frag_path: &str,
    out_mwa: &Path,
    src_hash: u64,
    fw: u32,
    fh: u32,
    fps: u32,
    frames: u32,
) -> i32 {
    let n = frames.clamp(12, 120) as usize;
    let overlap = (n / 4).clamp(6, 16);
    let dt = 1.0f32 / (fps.max(1) as f32);
    let t0 = 2.0f32; // start with all particles/rain/steam already in flight
    let w = fw as usize;
    let h = fh as usize;
    let frame_bytes = w * h * 3;

    unsafe {
        let Some(r) = OffscreenRenderer::new(img_path, frag_path, fw as i32, fh as i32) else {
            return 1;
        };

        let mut scratch = vec![0u8; frame_bytes];
        // Render head overlap frames (0..overlap) first so we can dissolve tail into them.
        let mut head_raw: Vec<Vec<u8>> = Vec::with_capacity(overlap);
        for f in 0..overlap {
            let mut buf = vec![0u8; frame_bytes];
            r.render_rgb_topdown(t0 + (f as f32) * dt, &mut scratch, &mut buf);
            head_raw.push(buf);
        }

        // Render middle steady frames (overlap..n) and tail overlap frames (n..n+overlap).
        // We stream encoding directly so peak RAM stays tiny (~15 MiB max during bake!).
        let mut tail_raw: Vec<Vec<u8>> = Vec::with_capacity(overlap);
        for f in 0..overlap {
            let mut buf = vec![0u8; frame_bytes];
            r.render_rgb_topdown(t0 + ((n + f) as f32) * dt, &mut scratch, &mut buf);
            tail_raw.push(buf);
        }

        // Build seamlessly dissolved frames 0..overlap:
        // At f=0, frac = 1/(overlap+1) -> mostly tail[0] (= frame n, seamless from frame n-1!).
        // At f=overlap-1, frac = overlap/(overlap+1) -> mostly head[overlap-1] (seamless into frame overlap!).
        let mut blended_head: Vec<Vec<u8>> = Vec::with_capacity(overlap);
        for f in 0..overlap {
            let mut out = vec![0u8; frame_bytes];
            let frac = ((f + 1) as f32) / ((overlap + 1) as f32);
            bayer_temporal_dissolve(&head_raw[f], &tail_raw[f], w, h, frac, &mut out);
            blended_head.push(out);
        }
        drop(head_raw);
        drop(tail_raw);

        let base_rgb = blended_head[0].clone();
        let mut deltas: Vec<(Vec<u8>, u32, u16, u16, u16, u16, [u16; TILE_ROWS])> =
            Vec::with_capacity(n);
        // Placeholder for frame 0 wrap delta (frame n-1 -> frame 0), filled at the end.
        deltas.push((Vec::new(), 0, 0, 0, 0, 0, [0u16; TILE_ROWS]));

        let mut raw_delta = Vec::with_capacity(frame_bytes / 2);
        let mut prev = blended_head[0].clone();

        // Encode frames 1..overlap from blended_head.
        for f in 1..overlap {
            let cur = &blended_head[f];
            if let Some(m) = pack_planar_delta(&prev, cur, w, h, &mut raw_delta) {
                let comp = lz4_compress(&raw_delta);
                deltas.push((comp, raw_delta.len() as u32, m.x0, m.y0, m.x1, m.y1, m.tile_mask));
            } else {
                deltas.push((Vec::new(), 0, 0, 0, 0, 0, [0u16; TILE_ROWS]));
            }
            prev.copy_from_slice(cur);
        }
        drop(blended_head);

        // Stream-render and encode frames overlap..n on the fly (only 2 RGB frames in RAM!).
        let mut cur = vec![0u8; frame_bytes];
        for f in overlap..n {
            r.render_rgb_topdown(t0 + (f as f32) * dt, &mut scratch, &mut cur);
            if let Some(m) = pack_planar_delta(&prev, &cur, w, h, &mut raw_delta) {
                let comp = lz4_compress(&raw_delta);
                deltas.push((comp, raw_delta.len() as u32, m.x0, m.y0, m.x1, m.y1, m.tile_mask));
            } else {
                deltas.push((Vec::new(), 0, 0, 0, 0, 0, [0u16; TILE_ROWS]));
            }
            std::mem::swap(&mut prev, &mut cur);
        }

        // Encode seamless wrap delta: frame (n-1) -> frame 0 (`base_rgb`)!
        if let Some(m) = pack_planar_delta(&prev, &base_rgb, w, h, &mut raw_delta) {
            let comp = lz4_compress(&raw_delta);
            deltas[0] = (comp, raw_delta.len() as u32, m.x0, m.y0, m.x1, m.y1, m.tile_mask);
        }

        match write_mwa_atomic(out_mwa, fw, fh, fps, src_hash, &base_rgb, &deltas) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("misa-wall baker: {e}");
                1
            }
        }
    }
}
