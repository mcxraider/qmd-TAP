import { afterEach, describe, expect, test, vi } from "vitest";
import { MissingTypeSafeKeyError, TypeSafeLLM } from "../src/llm-typesafe.js";
import type { LLM } from "../src/llm.js";

const base = {} as LLM;
const previousKey = process.env.TYPESAFE_API_KEY;
afterEach(() => {
  vi.restoreAllMocks();
  if (previousKey === undefined) delete process.env.TYPESAFE_API_KEY;
  else process.env.TYPESAFE_API_KEY = previousKey;
});

describe("TypeSafe reranking", () => {
  test("requires a key for reranked queries", async () => {
    delete process.env.TYPESAFE_API_KEY;
    await expect(new TypeSafeLLM(base).rerank("query", [{ file: "a", text: "text" }]))
      .rejects.toBeInstanceOf(MissingTypeSafeKeyError);
  });

  test("scores each passage and preserves its source index", async () => {
    process.env.TYPESAFE_API_KEY = "test-key";
    const fetchMock = vi.fn(async (_url: string | URL | Request, init?: RequestInit) => {
      const body = JSON.parse(init?.body as string);
      const score = body.state.passage === "answer" ? 0.9 : 0.1;
      return new Response(JSON.stringify({ answers: { relevant: { type: "noul", noul: score } } }));
    });
    vi.spyOn(globalThis, "fetch").mockImplementation(fetchMock);
    const result = await new TypeSafeLLM(base).rerank("question", [
      { file: "a", text: "other" }, { file: "b", text: "answer" },
    ]);
    expect(result.results).toEqual([
      { file: "b", index: 1, score: 0.9 },
      { file: "a", index: 0, score: 0.1 },
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  test("rejects invalid scores so they cannot enter the cache", async () => {
    process.env.TYPESAFE_API_KEY = "test-key";
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => new Response(JSON.stringify({ answers: { relevant: { type: "noul", noul: 2 } } })));
    await expect(new TypeSafeLLM(base).rerank("query", [{ file: "a", text: "text" }]))
      .rejects.toThrow("invalid score");
  });
});

test("limits concurrent passages and truncates input", async () => {
  process.env.TYPESAFE_API_KEY = "test-key";
  let active = 0;
  let peak = 0;
  const requests: Array<{ state: { passage: string }; model: string }> = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (_url: string | URL | Request, init?: RequestInit) => {
    requests.push(JSON.parse(String(init?.body)));
    peak = Math.max(peak, ++active);
    await new Promise(resolve => setTimeout(resolve, 1));
    active--;
    return new Response(JSON.stringify({ answers: { relevant: { type: "noul", noul: 0.5 } } }));
  });
  const result = await new TypeSafeLLM(base).rerank("query", Array.from({ length: 20 }, (_, i) => ({ file: String(i), text: "x".repeat(9000) })));
  expect(peak).toBe(8);
  expect(requests.every(r => r.state.passage.length === 8000 && r.model === "jev-1.13.0")).toBe(true);
  expect(result.results.map(r => r.index)).toEqual(Array.from({ length: 20 }, (_, i) => i));
});

test("retries one transient response", async () => {
  process.env.TYPESAFE_API_KEY = "test-key";
  const mock = vi.fn().mockResolvedValueOnce(new Response("busy", { status: 503 }))
    .mockResolvedValueOnce(new Response(JSON.stringify({ answers: { relevant: { type: "noul", noul: 0.9 } } })));
  vi.spyOn(globalThis, "fetch").mockImplementation(mock);
  expect((await new TypeSafeLLM(base).rerank("query", [{ file: "a", text: "answer" }])).results[0]?.score).toBe(0.9);
  expect(mock).toHaveBeenCalledTimes(2);
});

test("aborts outstanding requests at the shared deadline", async () => {
  process.env.TYPESAFE_API_KEY = "test-key";
  vi.spyOn(globalThis, "fetch").mockImplementation((_url: string | URL | Request, init?: RequestInit) => new Promise((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => reject(new Error("deadline exceeded")), { once: true });
  }));
  await expect(new TypeSafeLLM(base).rerank("query", [{ file: "a", text: "answer" }])).rejects.toThrow("deadline exceeded");
});
