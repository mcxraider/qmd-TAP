import type { LLM, EmbedOptions, EmbeddingResult, GenerateOptions, GenerateResult, ModelInfo, Queryable, RerankDocument, RerankOptions, RerankResult } from "./llm.js";
import type { Token as LlamaToken } from "node-llama-cpp";

const MODEL = "jev-1.13.0";
const QUESTION_VERSION = "qmd-relevance-v1";
export const TYPESAFE_RERANK_MODEL = `typesafe:${MODEL}:${QUESTION_VERSION}`;
const DEADLINE_MS = 8_000;

export class MissingTypeSafeKeyError extends Error {
  constructor() { super("Reranking requires TYPESAFE_API_KEY"); }
}

/** Keep the existing embedding/generation backend; replace only its ranking head. */
export class TypeSafeLLM implements LLM {
  constructor(private readonly base: LLM) {}

  get embedModelName() { return this.base.embedModelName; }
  get generateModelName() { return this.base.generateModelName; }
  get rerankModelName() { return TYPESAFE_RERANK_MODEL; }
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> { return this.base.embed(text, options); }
  embedBatch(texts: string[], options?: EmbedOptions): Promise<(EmbeddingResult | null)[]> { return this.base.embedBatch(texts, options); }
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null> { return this.base.generate(prompt, options); }
  modelExists(model: string): Promise<ModelInfo> { return this.base.modelExists(model); }
  expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]> { return this.base.expandQuery(query, options); }
  tokenize(text: string): Promise<readonly LlamaToken[]> { return this.base.tokenize(text); }
  detokenize(tokens: readonly LlamaToken[]): Promise<string> { return this.base.detokenize(tokens); }
  dispose(): Promise<void> { return this.base.dispose(); }

  async rerank(query: string, documents: RerankDocument[], _options?: RerankOptions): Promise<RerankResult> {
    const key = process.env.TYPESAFE_API_KEY;
    if (!key) throw new MissingTypeSafeKeyError();
    if (documents.length === 0) return { results: [], model: this.rerankModelName };
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), DEADLINE_MS);
    const scores = new Array<number>(documents.length);
    let next = 0;
    try {
      await Promise.all(Array.from({ length: Math.min(8, documents.length) }, async () => {
        while (next < documents.length) {
          const i = next++;
          const document = documents[i]!;
          const request = {
            model: MODEL,
            state: { query, passage: document.text.slice(0, 8_000) },
            questions: { relevant: {
              type: "noul",
              instructions: "Does the passage directly answer or provide useful evidence for the query?",
              criteria: {
                true: "It contains information that helps answer the specific query, including a relevant explanation or example.",
                false: "It merely shares a topic or keywords, or does not help answer the query.",
              },
            } },
          };
          let response: Response;
          for (let attempt = 0; ; attempt++) {
            response = await fetch("https://api.typesafe.ai/v1/systemone", {
              method: "POST",
              headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
              body: JSON.stringify(request),
              signal: controller.signal,
            });
            if (![429, 503, 529].includes(response.status) || attempt >= 1) break;
            await new Promise<void>((resolve, reject) => {
              const wait = setTimeout(resolve, 250);
              controller.signal.addEventListener("abort", () => { clearTimeout(wait); reject(controller.signal.reason); }, { once: true });
            });
          }
          if (!response.ok) throw new Error(`TypeSafe reranking failed: HTTP ${response.status}`);
          const body = await response.json() as { answers?: { relevant?: { type?: string; noul?: number } } };
          const score = body.answers?.relevant?.noul;
          if (body.answers?.relevant?.type !== "noul" || typeof score !== "number" || !Number.isFinite(score) || score < 0 || score > 1) {
            throw new Error("TypeSafe reranking returned an invalid score");
          }
          scores[i] = score;
        }
      }));
      return {
        model: this.rerankModelName,
        results: documents.map((document, index) => ({ file: document.file, index, score: scores[index]! }))
          .sort((a, b) => b.score - a.score || a.index - b.index),
      };
    } finally {
      clearTimeout(timer);
      controller.abort();
    }
  }
}
