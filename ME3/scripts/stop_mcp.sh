#!/usr/bin/env bash
pkill -f "mcp_servers\." || true; sudo pkill -f "mcp_servers.home_server" || true
