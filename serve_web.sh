#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# serve_web.sh — public https URL for the Streamlit app via a Cloudflare tunnel
#
#   * Starts Streamlit (0.0.0.0) ONLY if it isn't already running on the port.
#   * Opens a Cloudflare quick tunnel with retries.
#   * A tunnel failure NEVER kills Streamlit (run it in its own pane safely).
#
#   Usage (box, from /workspace/EMBRIO):   bash serve_web.sh [port]
# ---------------------------------------------------------------------------
set -u
PORT="${1:-8501}"

# --- cloudflared (install once) --------------------------------------------
if ! command -v cloudflared >/dev/null 2>&1; then
  echo "==> Installing cloudflared ..."
  curl -fsSL -o /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x /usr/local/bin/cloudflared
fi

# --- ensure Streamlit is up (don't double-start) ----------------------------
if ! curl -s -o /dev/null "http://localhost:$PORT"; then
  echo "==> Starting Streamlit on :$PORT (logs: /tmp/streamlit.log) ..."
  nohup streamlit run app.py \
    --server.port "$PORT" --server.address 0.0.0.0 --server.headless true \
    --server.enableCORS false --server.enableXsrfProtection false \
    >/tmp/streamlit.log 2>&1 &
  until curl -s -o /dev/null "http://localhost:$PORT"; do sleep 1; done
fi
echo "==> Streamlit is up on :$PORT"

# --- public tunnel, with retries (does NOT kill Streamlit on failure) -------
echo
echo "==================================================================="
echo "  Watch for:  https://<name>.trycloudflare.com   (your public link)"
echo "==================================================================="
for i in 1 2 3 4 5; do
  echo "==> tunnel attempt $i ..."
  cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" && break
  echo "==> tunnel failed; retrying in 4s (Streamlit stays up) ..."
  sleep 4
done
echo "If the tunnel keeps failing, Cloudflare may be blocked from this host —"
echo "use Vast's own port mapping instead (see the notes I sent)."
