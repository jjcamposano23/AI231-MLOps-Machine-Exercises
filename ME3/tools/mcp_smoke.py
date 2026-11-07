"""Smoke test: start the MCP servers (if needed) and call a few tools over MCP."""
import asyncio, sys, time
sys.path.insert(0, ".")
from assistant.config import load
from assistant.mcp_link import McpLink, ensure_servers, stop_children

async def main():
    cfg = load()
    urls = {"asr": cfg.asr_mcp_url, "tts": cfg.tts_mcp_url, **cfg.tool_mcp_urls, "agent": cfg.agent_mcp_url}
    ensure_servers(urls)
    links = {n: McpLink(u, n) for n, u in urls.items()}
    for n, l in links.items():
        await l.start()
    checks = [("asr", "info", {}), ("tts", "info", {}), ("assist", "get_time", {}),
              ("assist", "get_weather", {"city": "Quezon City"}),
              ("assist", "web_search", {"query": "Raspberry Pi 5 price Philippines", "max_results": 2}),
              ("assist", "set_timer", {"seconds": 2, "label": "tea"}),
              ("home", "set_light_color", {"color": "blue"}), ("home", "get_home_state", {})]
    for n, tool, args in checks:
        t0 = time.monotonic()
        try:
            r = await links[n].call(tool, args)
            print(f"OK  {n}.{tool} {round((time.monotonic()-t0)*1000)} ms -> {str(r)[:160]}")
        except Exception as e:
            print(f"ERR {n}.{tool}: {e!r}")
    await asyncio.sleep(2.5)
    print("due:", await links["assist"].call("internal_pop_due"))
    for l in links.values():
        await l.close()

try:
    asyncio.run(main())
finally:
    if "--keep" not in sys.argv:
        stop_children()
