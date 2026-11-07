#!/usr/bin/env bash
# Start the 5 MCP servers on the Pi in the background (logs in logs/). The home server runs with sudo so the
# WS2812 strip works; the assistant itself stays unprivileged (sudo breaks the mic).
cd "$(dirname "$0")/.."
mkdir -p logs
for s in asr tts assist agent; do
  nohup .venv/bin/python -m mcp_servers.${s}_server > logs/mcp_${s}.log 2>&1 &
done
if [ "${HOME_SUDO:-1}" = "1" ]; then
  sudo -E nohup .venv/bin/python -m mcp_servers.home_server > logs/mcp_home.log 2>&1 &
else
  nohup .venv/bin/python -m mcp_servers.home_server > logs/mcp_home.log 2>&1 &
fi
sleep 3; ss -ltnp 2>/dev/null | grep -E ':810[1-5]|:8080' || true
