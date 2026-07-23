//! Rust <-> xterm.js bridge. All xterm logic lives in `assets/corrode-term.js`;
//! this just binds the three shim functions and adapts them to Rust closures.

use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::wasm_bindgen;
use web_sys::HtmlElement;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_name = corrodeTermInit)]
    fn term_init(
        mount: &HtmlElement,
        on_data: &Closure<dyn FnMut(String)>,
        on_resize: &Closure<dyn FnMut(u32, u32)>,
    );

    #[wasm_bindgen(js_name = corrodeTermWrite)]
    fn term_write(data: &[u8]);
}

/// Write pty output bytes into the terminal.
pub fn write(data: &[u8]) {
    term_write(data);
}

/// Mount an xterm terminal on `mount`. `on_data` receives typed input (send as
/// `TerminalInput`); `on_resize` receives `(cols, rows)` (send as `TerminalResize`).
/// The closures live for the page's lifetime (`forget`).
pub fn init(
    mount: HtmlElement,
    on_data: impl FnMut(String) + 'static,
    on_resize: impl FnMut(u32, u32) + 'static,
) {
    let data_cb = Closure::wrap(Box::new(on_data) as Box<dyn FnMut(String)>);
    let resize_cb = Closure::wrap(Box::new(on_resize) as Box<dyn FnMut(u32, u32)>);
    term_init(&mount, &data_cb, &resize_cb);
    data_cb.forget();
    resize_cb.forget();
}
