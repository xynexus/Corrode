//! Real pty-backed terminal sessions — the daemon's side of the virtual terminal.
//!
//! Each session is an OS pty running a shell (`portable-pty`). `TerminalInput`
//! bytes are written to the pty; the pty's output streams back as `TerminalOutput`.
//! `TerminalResize` sets the pty geometry so full-screen TUIs render correctly.
//!
//! portable-pty is blocking, so a per-session **reader thread** pumps pty output
//! into the async event channel via `blocking_send` (safe: it's a plain OS thread,
//! not a tokio worker). Sessions are shared, tmux-style: every `ensure` touch
//! re-points the session's output at the touching connection, so a page reload
//! (new ws, same session id) adopts the running shell instead of orphaning it.
//! Output with no listener is dropped; the shell dies only on EOF (exit).

use corrode_core::AgentEvent;
use portable_pty::{native_pty_system, CommandBuilder, MasterPty, PtySize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::Sender;

struct Session {
    master: Box<dyn MasterPty + Send>, // for resize
    writer: Box<dyn Write + Send>,     // for input
    /// Where this session's output currently goes: the last connection to touch
    /// it. Re-pointed on every `ensure`, so a reloaded page adopts the shell.
    events: Sender<AgentEvent>,
    /// Current pty geometry, so a same-size resize (a reloaded page re-attaching)
    /// can be turned into a redraw nudge instead of a silent no-op.
    size: PtySize,
}

type SessionMap = Arc<Mutex<HashMap<String, Session>>>;

/// The daemon's live terminal sessions, keyed by client-chosen session id. The map
/// is shared with each session's reader thread so it can evict itself on exit.
pub struct Terminals {
    sessions: SessionMap,
    /// Where spawned shells start: the daemon's repo root (`CORRODE_REPO`).
    cwd: std::path::PathBuf,
    /// Optional bubblewrap confinement for the interactive shell (default off).
    sandbox: crate::sandbox::Sandbox,
}

impl Terminals {
    pub fn new(cwd: std::path::PathBuf) -> Self {
        Self {
            sessions: SessionMap::default(),
            cwd,
            sandbox: crate::sandbox::Sandbox::disabled(),
        }
    }

    /// Confine spawned shells with this sandbox (builder; default is disabled).
    pub fn with_sandbox(mut self, sandbox: crate::sandbox::Sandbox) -> Self {
        self.sandbox = sandbox;
        self
    }

    /// Spawn a pty+shell for `id` if absent; either way, point the session's
    /// output at `events` (the calling connection adopts the session). Returns
    /// true when an existing session changed hands to a *different* connection.
    fn ensure(&self, id: &str, events: &Sender<AgentEvent>, size: PtySize) -> anyhow::Result<bool> {
        let mut map = self.sessions.lock().unwrap();
        if let Some(s) = map.get_mut(id) {
            // ponytail: latest toucher wins — two live tabs on one session id
            // flap output between them; per-tab session ids if that ever matters.
            let adopted = !s.events.same_channel(events);
            s.events = events.clone();
            return Ok(adopted);
        }
        let pair = native_pty_system().openpty(size)?;
        // Interactive, NON-login shell: sources ~/.bashrc but not /etc/profile.d,
        // whose 80-systemd-osc-context.sh emits an OSC 3008 sequence xterm.js can't
        // parse (it printed as noise and garbled input). Env is inherited from the
        // daemon, so PATH/venv survive. TERM advertises a type xterm understands.
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());
        // sandbox.wrap is a no-op when disabled: plain `<shell> -i`. When on, it
        // becomes `bwrap … -- <shell> -i`; env (incl. TERM below) passes through to
        // the child since we don't --clearenv, and --chdir sets the inner cwd.
        let (prog, args) = self.sandbox.wrap(&self.cwd, &[shell.as_str(), "-i"]);
        let mut cmd = CommandBuilder::new(prog);
        for arg in &args {
            cmd.arg(arg);
        }
        cmd.env("TERM", "xterm-256color");
        cmd.cwd(&self.cwd);
        let child = pair.slave.spawn_command(cmd)?;
        drop(pair.slave); // so the master read hits EOF when the shell exits
        let mut reader = pair.master.try_clone_reader()?;
        let writer = pair.master.take_writer()?;

        let session_id = id.to_string(); // moved into the reader thread
        let sessions = self.sessions.clone();
        std::thread::spawn(move || {
            let mut child = child; // owned here -> killed on thread exit
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break, // EOF or error
                    Ok(n) => {
                        // Send to the session's *current* owner (clone under the
                        // lock, send outside it — blocking_send can park). A dead
                        // owner just drops the chunk; the shell lives on and the
                        // next client to touch the session adopts it.
                        let tx = sessions
                            .lock()
                            .unwrap()
                            .get(&session_id)
                            .map(|s: &Session| s.events.clone());
                        let Some(tx) = tx else { break };
                        let ev = AgentEvent::TerminalOutput {
                            session: session_id.clone(),
                            data: buf[..n].to_vec(),
                        };
                        let _ = tx.blocking_send(ev);
                    }
                }
            }
            let _ = child.kill();
            let _ = child.wait(); // reap — kill alone leaves a zombie
            sessions.lock().unwrap().remove(&session_id);
        });

        map.insert(
            id.to_string(),
            Session { master: pair.master, writer, events: events.clone(), size },
        );
        Ok(false)
    }

    /// Write keystrokes to the session's pty (opening it at a default size if new).
    pub fn input(&self, id: &str, data: &[u8], events: &Sender<AgentEvent>) -> anyhow::Result<()> {
        self.ensure(id, events, default_size())?;
        let mut map = self.sessions.lock().unwrap();
        if let Some(s) = map.get_mut(id) {
            s.writer.write_all(data)?;
            s.writer.flush()?;
        }
        Ok(())
    }

    /// Set the pty geometry (opening the session if this is the first message).
    pub fn resize(
        &self,
        id: &str,
        cols: u16,
        rows: u16,
        events: &Sender<AgentEvent>,
    ) -> anyhow::Result<()> {
        let size = PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        };
        let adopted = self.ensure(id, events, size)?;
        let mut map = self.sessions.lock().unwrap();
        if let Some(s) = map.get_mut(id) {
            s.master.resize(size)?;
            // A reload re-attaches at the same size, so no SIGWINCH fires and the
            // shell never repaints — the new tab would sit blank. Ctrl-L asks
            // readline (or a TUI) to redraw the screen. ponytail: mid-`cat` the
            // byte lands in stdin; harmless enough for a dev terminal.
            if adopted && s.size == size {
                s.writer.write_all(b"\x0c")?;
                s.writer.flush()?;
            }
            s.size = size;
        }
        Ok(())
    }
}

fn default_size() -> PtySize {
    PtySize {
        rows: 24,
        cols: 80,
        pixel_width: 0,
        pixel_height: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Spawns a real shell over a pty and confirms its output streams back through
    // the event channel — the actual "real terminal" path, no echo.
    #[tokio::test]
    async fn pty_runs_a_shell_and_streams_output() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(64);
        let terms = Terminals::new(env!("CARGO_MANIFEST_DIR").into());
        terms.resize("t", 80, 24, &tx).unwrap(); // opens the pty + shell
        terms.input("t", b"echo corrode-ok && pwd\n", &tx).unwrap();

        let mut seen = Vec::new();
        let found = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(ev) = rx.recv().await {
                if let AgentEvent::TerminalOutput { data, .. } = ev {
                    seen.extend_from_slice(&data);
                    let text = String::from_utf8_lossy(&seen);
                    // marker proves the shell ran; pwd proves it started in `cwd`
                    if text.contains("corrode-ok") && text.contains(env!("CARGO_MANIFEST_DIR")) {
                        return true;
                    }
                }
            }
            false
        })
        .await
        .unwrap_or(false);

        assert!(
            found,
            "shell should echo the marker from the given cwd; got: {}",
            String::from_utf8_lossy(&seen)
        );
    }

    // A second connection (page reload) touching the same session id adopts the
    // running shell: output flows to the new channel, the shell isn't respawned.
    #[tokio::test]
    async fn reconnect_adopts_the_running_session() {
        let (tx1, mut rx1) = tokio::sync::mpsc::channel(64);
        let terms = Terminals::new(env!("CARGO_MANIFEST_DIR").into());
        terms.resize("r", 80, 24, &tx1).unwrap();
        // wait for the first prompt so the shell is up
        tokio::time::timeout(std::time::Duration::from_secs(5), rx1.recv())
            .await
            .expect("shell should print a prompt")
            .expect("channel open");

        drop(rx1); // "tab closed"
        let (tx2, mut rx2) = tokio::sync::mpsc::channel(64);
        terms.input("r", b"echo adopted-ok\n", &tx2).unwrap(); // new tab touches it

        let mut seen = Vec::new();
        let found = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(ev) = rx2.recv().await {
                if let AgentEvent::TerminalOutput { data, .. } = ev {
                    seen.extend_from_slice(&data);
                    if String::from_utf8_lossy(&seen).contains("adopted-ok") {
                        return true;
                    }
                }
            }
            false
        })
        .await
        .unwrap_or(false);

        assert!(
            found,
            "new channel should receive the adopted shell's output; got: {}",
            String::from_utf8_lossy(&seen)
        );
    }
}
