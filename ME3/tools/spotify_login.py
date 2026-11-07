"""One-time Spotify sign-in for Iris (PKCE: needs only SPOTIFY_CLIENT_ID; the token is cached in data/).

Run it on the Pi through an SSH tunnel, so the browser on your laptop can reach the Pi's callback port:
    ssh -L 8888:127.0.0.1:8888 -t jjcamposano@vcm.local "cd ~/ME3 && .venv/bin/python -m tools.spotify_login"
Open the printed link on the laptop, approve, and the page should say "Iris is connected to Spotify".
If the page does not load, copy the full address from the browser bar and paste it here instead.
"""
from __future__ import annotations

import http.server
import os
import sys
import threading
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, ".")
from mcp_servers.common import load_env  # noqa: E402

load_env()
from mcp_servers.spotify_player import CACHE, REDIRECT, auth_manager  # noqa: E402

got: dict = {}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        q = parse_qs(urlparse(self.path).query)
        got.update({k: v[0] for k, v in q.items()})
        ok = "code" in q
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        msg = "Iris is connected to Spotify. You can close this tab." if ok else f"Spotify said: {q.get('error')}"
        self.wfile.write(f"<h2 style='font-family:sans-serif'>{msg}</h2>".encode())
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, *a):
        pass


def main() -> None:
    if not os.environ.get("SPOTIFY_CLIENT_ID"):
        sys.exit("Add SPOTIFY_CLIENT_ID=... to ~/ME3/.env first (from developer.spotify.com/dashboard).")
    am = auth_manager()
    url = am.get_authorize_url()
    port = urlparse(REDIRECT).port or 8888
    print("\n1) Open this link in your laptop's browser and click Agree:\n\n" + url + "\n")
    print(f"2) Waiting for Spotify to redirect to {REDIRECT} (through the SSH tunnel)...")
    print("   If the page doesn't load, paste the full address from the browser bar here and press Enter.\n")
    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    def paste():
        try:
            line = input().strip()
        except EOFError:
            return
        if line:
            got.update({k: v[0] for k, v in parse_qs(urlparse(line).query).items()})
            srv.shutdown()
    threading.Thread(target=paste, daemon=True).start()
    while "code" not in got and "error" not in got:
        threading.Event().wait(0.3)
    if "error" in got:
        sys.exit(f"Spotify sign-in failed: {got['error']}")
    am.get_access_token(got["code"], check_cache=False)
    os.chmod(CACHE, 0o600)
    import spotipy
    me = spotipy.Spotify(auth_manager=am).current_user()
    print(f"\nSigned in to Spotify as {me.get('display_name') or me.get('id')} "
          f"({me.get('product', '?')} account). Token saved to {CACHE}.")
    if me.get("product") != "premium":
        print("NOTE: playback control needs Spotify Premium.")


if __name__ == "__main__":
    main()
