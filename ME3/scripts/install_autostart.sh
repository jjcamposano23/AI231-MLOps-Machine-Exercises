#!/usr/bin/env bash
# Start Iris (and her 5 MCP servers) automatically at every boot, as user services (no sudo).
# Needs desktop auto-login (Raspberry Pi OS default), which starts the user's systemd at boot.
#   bash ~/ME3/scripts/install_autostart.sh            install + start now
#   bash ~/ME3/scripts/install_autostart.sh --remove   uninstall
set -e
cd "$(dirname "$0")/.."
mkdir -p ~/.config/systemd/user logs
if [ "${1:-}" = "--remove" ]; then
  systemctl --user disable --now iris.service 2>/dev/null || true
  rm -f ~/.config/systemd/user/iris.service ~/.config/systemd/user/iris-mcp@.service
  systemctl --user daemon-reload
  echo "autostart removed"; exit 0
fi
pkill -f "[a]ssistant.main" || true                 # stop manually started copies (they hold the mic / ports)
pkill -f "[m]cp_servers\.(asr|tts|assist|home|agent)_server" || true
sleep 1
cp scripts/systemd/iris.service scripts/systemd/iris-mcp@.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable iris.service
systemctl --user restart iris.service
echo "Iris autostart installed. Status: systemctl --user status iris    Log: tail -f ~/ME3/logs/iris.log"
