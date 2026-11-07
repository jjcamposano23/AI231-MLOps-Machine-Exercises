"""Text-only test of Iris's brain: same LLM, system prompt and MCP tools as the voice pipeline, no audio.

    python tools/text_chat.py "what's the weather in Davao?" "turn the lights blue" ...
    python tools/text_chat.py            (interactive)
"""
from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, ".")
from pipecat.frames.frames import (FunctionCallInProgressFrame, LLMFullResponseEndFrame, LLMFullResponseStartFrame,
                                   LLMRunFrame, LLMTextFrame)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

from assistant.config import load
from assistant.main import Assistant, setup_logging
from assistant.mcp_link import ensure_servers, stop_children
from assistant.observers import EmulatorObserver
from mcp_servers import emulator_bus


class Printer(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.buf, self.idle, self.t0, self.first, self.tools = [], asyncio.Event(), 0.0, None, 0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self.idle.clear()
        elif isinstance(frame, LLMTextFrame):
            if self.first is None:
                self.first = time.monotonic() - self.t0
            self.buf.append(frame.text)
        elif isinstance(frame, FunctionCallInProgressFrame):
            self.tools += 1
            print(f"   [tool] {frame.function_name}({frame.arguments})")
        elif isinstance(frame, LLMFullResponseEndFrame):
            self.idle.set()
        await self.push_frame(frame, direction)


async def main(prompts):
    cfg = load()
    setup_logging("WARNING")
    use_agent = "--agent" in sys.argv
    urls = dict(cfg.tool_mcp_urls)
    if use_agent:
        urls["agent"] = cfg.agent_mcp_url
    ensure_servers(urls)
    app = Assistant(cfg)
    if use_agent:
        from assistant.mcp_link import McpLink
        app.links["agent"] = McpLink(cfg.agent_mcp_url, "agent")
    llm, context, tools = await app.llm_and_context(use_agent)
    print(f"LLM {cfg.llm_provider}/{cfg.llm_model}, {len(tools)} tools")
    user_agg, asst_agg = LLMContextAggregatorPair(context)
    pr = Printer()
    worker = PipelineWorker(Pipeline([user_agg, llm, pr, asst_agg]), params=PipelineParams(),
                            observers=[EmulatorObserver("text")])      # typed turns show in the web emulator too
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    run = asyncio.create_task(runner.run())
    await asyncio.sleep(0.5)

    async def ask(text):
        pr.buf, pr.first, pr.t0, pr.tools = [], None, time.monotonic(), 0
        context.add_message({"role": "user", "content": text})
        emulator_bus.emit("user", text=text, session="text", typed=True)
        pr.idle.clear()
        await worker.queue_frames([LLMRunFrame()])
        while True:                                     # wait for the final answer (after any tool round trips)
            await asyncio.wait_for(pr.idle.wait(), 120)
            await asyncio.sleep(1.5)
            if pr.idle.is_set() and (pr.buf or not pr.tools or time.monotonic() - pr.t0 > 60):
                break                                   # a tool call with no text yet: its answer is still coming
            pr.idle.clear()
        ttft = f"{pr.first:.2f}s" if pr.first is not None else "-"
        print(f"you > {text}\nIris> {''.join(pr.buf).strip()}   (first token {ttft}, "
              f"total {time.monotonic() - pr.t0:.1f}s)\n")

    if prompts:
        for p in prompts:
            await ask(p)
        if use_agent:
            print("waiting up to 5 min for the background agent ...")
            pr.idle.clear()
            try:
                await asyncio.wait_for(pr.idle.wait(), 300)
                await asyncio.sleep(2)
                print(f"Iris> {''.join(pr.buf).strip()}")
            except asyncio.TimeoutError:
                print("agent still running")
    else:
        while (line := await asyncio.to_thread(input, "you > ")).strip():
            await ask(line)
    await worker.cancel()
    run.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main([a for a in sys.argv[1:] if not a.startswith("--")]))
    finally:
        stop_children()
