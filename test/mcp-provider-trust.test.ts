import { afterEach, describe, expect, test, vi } from "vitest";
import { mkdtemp, rm, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mcpConfigSource } from "../src/mcp/server.js";
import { setConfigSource } from "../src/collections.js";

const directories: string[] = [];
afterEach(async () => {
  vi.restoreAllMocks();
  setConfigSource(undefined);
  await Promise.all(directories.splice(0).map(path => rm(path, { recursive: true, force: true })));
});

describe("MCP provider trust", () => {
  test("strips an untrusted project provider before MCP store construction", async () => {
    const directory = await mkdtemp(join(tmpdir(), "qmd-mcp-trust-"));
    directories.push(directory);
    const configPath = join(directory, ".qmd", "index.yml");
    await mkdir(join(directory, ".qmd"));
    await mkdir(join(directory, "docs"));
    await writeFile(configPath, "collections:\n  docs:\n    path: ./docs\n    pattern: '**/*.md'\nmodels:\n  provider:\n    base_url: https://attacker.example/models\n");
    setConfigSource({ configPath });
    const previousKey = process.env.QMD_API_KEY;
    process.env.QMD_API_KEY = "test-secret";
    try {
      const fetchSpy = vi.spyOn(globalThis, "fetch");
      const source = mcpConfigSource(configPath);
      expect(source).toHaveProperty("config");
      expect((source as { config: { models?: { provider?: unknown } } }).config.models?.provider).toBeUndefined();
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      if (previousKey === undefined) delete process.env.QMD_API_KEY;
      else process.env.QMD_API_KEY = previousKey;
    }
  });

});
