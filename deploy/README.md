# Running Corrode as a service

Corrode is two processes: the **daemon** (stateful — swarm, VFS, embedded
HelixDB) and the **web** frontend (stateless — serves the wasm UI, proxies
`/agent` to the daemon). These are systemd **user** units so the daemon runs as
your login user (the sandbox needs an unprivileged user namespace, not root).

## 1. Build

```bash
# wasm bundle first — corrode-web embeds webui/dist at compile time
(cd webui && trunk build --release)

# daemon with the real store + doc ingestion (needs OpenSSL + pkg-config, see CLAUDE.md)
OPENSSL_LIB_DIR=/usr/lib/x86_64-linux-gnu OPENSSL_INCLUDE_DIR=/usr/include \
  cargo build -p corrode-daemon --release --features helix,docling
cargo build -p corrode-web --release
```

## 2. Configure

```bash
cp deploy/corrode.env.example deploy/corrode.env   # gitignored; edit it
```

Set at least `CORRODE_REPO`. For a shared host, also turn on `CORRODE_SANDBOX`
and point `CORRODE_USERS` at a users file (see the example). Run
`target/release/corrode-daemon doctor` to check host readiness (hipfire
reachable, sandbox working, users file valid) before starting the service.

## 3. Install & start

```bash
mkdir -p ~/.config/systemd/user
ln -s ~/Corrode/deploy/corrode-daemon.service ~/.config/systemd/user/
ln -s ~/Corrode/deploy/corrode-web.service    ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now corrode-web.service   # pulls in the daemon
# survive logout / start at boot:
loginctl enable-linger "$USER"
```

`corrode-web` `Wants=` the daemon, so enabling it starts both. Logs:
`journalctl --user -u corrode-daemon -f`. Open the UI at
`http://<host>:8787`.

> Paths in the units use `%h` (your home). If Corrode lives elsewhere, edit the
> `WorkingDirectory`/`ExecStart`/`EnvironmentFile` lines or override with
> `systemctl --user edit`.
