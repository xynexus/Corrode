// Thin wrapper around xterm.js, exposing a minimal surface the Rust/wasm side
// calls via wasm-bindgen. Keeps all xterm specifics in JS; Rust just pipes bytes.
//
// Globals (from the vendored UMD builds loaded before this): Terminal, FitAddon,
// WebglAddon. We tolerate either the namespace (FitAddon.FitAddon) or a direct
// class export.
(function () {
  var term = null;
  var fit = null;

  function cls(g, name) {
    return (g && g[name]) || g;
  }

  // corrodeTermInit(mountEl, onData, onResize)
  //   onData(string)      -> keystrokes to send to the daemon pty (TerminalInput)
  //   onResize(cols, rows) -> new geometry to send (TerminalResize)
  window.corrodeTermInit = function (mountEl, onData, onResize) {
    var Term = window.Terminal;
    var Fit = cls(window.FitAddon, "FitAddon");
    var Webgl = cls(window.WebglAddon, "WebglAddon");

    term = new Term({
      convertEol: false,
      cursorBlink: true,
      fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
      fontSize: 13,
      theme: { background: "#0c0d10", foreground: "#d7dbe0" },
    });
    fit = new Fit();
    term.loadAddon(fit);
    term.open(mountEl);
    // WebGL renderer is an optimization; fall back silently to the default
    // (canvas/DOM) renderer when WebGL is unavailable.
    try {
      term.loadAddon(new Webgl());
    } catch (e) {
      console.warn("[corrode] xterm WebGL addon unavailable, using default renderer:", e);
    }

    // Swallow systemd's OSC 3008 "context" sequence (emitted by login shells via
    // /etc/profile.d) — xterm.js doesn't recognize it and would print the payload.
    if (term.parser && term.parser.registerOscHandler) {
      term.parser.registerOscHandler(3008, function () { return true; });
    }

    term.onData(function (d) { onData(d); });

    function report() {
      try { fit.fit(); } catch (e) {}
      onResize(term.cols, term.rows);
    }
    report();
    // Keep the pty geometry in sync with the element size.
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(function () { report(); }).observe(mountEl);
    }
    term.focus();
  };

  // corrodeTermWrite(Uint8Array) — pty output bytes to render.
  window.corrodeTermWrite = function (data) {
    if (term) term.write(data);
  };
})();
