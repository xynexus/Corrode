//! Real pty-backed terminal sessions — the daemon's side of the virtual terminal.
//!
//! Each session is an OS pty running a shell (`portable-pty`). `TerminalInput`
//! bytes are written to the pty; the pty's output streams back as `TerminalOutput`.
//! `TerminalResize` sets the pty geometry so full-screen TUIs render correctly.
//!
//! portable-pty is blocking, so a per-session **reader thread** pumps pty output
//! into the async event channel via `blocking_send` (safe: it's a plain OS thread,
//! not a tokio worker). The thread owns the child, so the shell is killed when the
//! client disconnects (the send fails) or the shell exits (EOF).

use corrode_core::AgentEvent;
use portable_pty::{native_pty_system, CommandBuilder, MasterPty, PtySize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::Sender;

struct Session {
    master: Box<dyn MasterPty + Send>, // for resize
    writer: Box<dyn Write + Send>,     // for input
}

type SessionMap = Arc<Mutex<HashMap<String, Session>>>;

/// The daemon's live terminal sessions, keyed by client-chosen session id. The map
/// is shared with each session's reader thread so it can evict itself on exit.
#[derive(Default)]
pub struct Terminals {
    sessions: SessionMap,
}

impl Terminals {
    pub fn new() -> Self {
        Self::default()
    }

    /// Spawn a pty+shell for `id` if absent, streaming its output to `events`.
    // ponytail: sessions aren't reaped from the map on disconnect (the shell is
    // killed, but the stale entry lingers) and a `exit`ed shell can't be respawned
    // under the same id — fine for the single-terminal scaffold; add lifecycle mgmt
    // when multi-session lands. cwd is the daemon's cwd; set it to CORRODE_REPO later.
    fn ensure(&self, id: &str, events: &Sender<AgentEvent>, size: PtySize) -> anyhow::Result<()> {
        let mut map = self.sessions.lock().unwrap();
        if map.contains_key(id) {
            return Ok(());
        }
        let pair = native_pty_system().openpty(size)?;
        // Interactive, NON-login shell: sources ~/.bashrc but not /etc/profile.d,
        // whose 80-systemd-osc-context.sh emits an OSC 3008 sequence xterm.js can't
        // parse (it printed as noise and garbled input). Env is inherited from the
        // daemon, so PATH/venv survive. TERM advertises a type xterm understands.
        let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());
        let mut cmd = CommandBuilder::new(shell);
        cmd.arg("-i");
        cmd.env("TERM", "xterm-256color");
        let child = pair.slave.spawn_command(cmd)?;
        drop(pair.slave); // so the master read hits EOF when the shell exits
        let mut reader = pair.master.try_clone_reader()?;
        let writer = pair.master.take_writer()?;

        let events = events.clone();
        let session_id = id.to_string(); // moved into the reader thread
        let sessions = self.sessions.clone();
        std::thread::spawn(move || {
            let mut child = child; // owned here -> killed on thread exit
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break, // EOF or error
                    Ok(n) => {
                        let ev = AgentEvent::TerminalOutput {
                            session: session_id.clone(),
                            data: buf[..n].to_vec(),
                        };
                        if events.blocking_send(ev).is_err() {
                            break; // client gone
                        }
                    }
                }
            }
            let _ = child.kill();
            // Evict the now-dead session so a reload/reconnect spawns a fresh pty
            // (the disconnect happens well before the reconnect, so no id race).
            sessions.lock().unwrap().remove(&session_id);
        });

        map.insert(id.to_string(), Session { master: pair.master, writer });
        Ok(())
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
        self.ensure(id, events, size)?;
        let map = self.sessions.lock().unwrap();
        if let Some(s) = map.get(id) {
            s.master.resize(size)?;
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
        let terms = Terminals::new();
        terms.resize("t", 80, 24, &tx).unwrap(); // opens the pty + shell
        terms.input("t", b"echo corrode-ok\n", &tx).unwrap();

        let mut seen = Vec::new();
        let found = tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while let Some(ev) = rx.recv().await {
                if let AgentEvent::TerminalOutput { data, .. } = ev {
                    seen.extend_from_slice(&data);
                    if String::from_utf8_lossy(&seen).contains("corrode-ok") {
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
            "shell should echo the marker; got: {}",
            String::from_utf8_lossy(&seen)
        );
    }
}
