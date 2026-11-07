import asyncio, base64, sys, time
sys.path.insert(0, ".")
from assistant.config import load
from assistant.mcp_link import McpLink, ensure_servers, stop_children
async def main():
    cfg = load(); ensure_servers({"tts": cfg.tts_mcp_url})
    l = McpLink(cfg.tts_mcp_url, "tts"); await l.start()
    for text in ["Hi, I'm Iris.", "The weather in Davao is partly cloudy at thirty one degrees, with a high of thirty three and a forty percent chance of rain this afternoon, so you might want an umbrella."]*2:
        t0 = time.monotonic(); first = None; n = 0; b = 0
        async for c in l.stream("synthesize_stream", {"text": text, "sample_rate": 24000}):
            if first is None: first = time.monotonic() - t0
            n += 1; b += len(base64.b64decode(c))
        print(f"{len(text):3d} chars: first chunk {first*1000:.0f} ms, {n} chunks, {b/48000:.2f}s audio, done {1000*(time.monotonic()-t0):.0f} ms")
    await l.close()
try: asyncio.run(main())
finally: stop_children()
