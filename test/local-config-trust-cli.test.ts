/**
 * End-to-end trust gate for project-local collection paths and model URIs (#889).
 *
 * Spawns real `qmd update` processes against a fixture that plays the part of a
 * freshly cloned repository shipping its own `.qmd/index.yml`.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const runnerArgs = isBunRuntime ? [qmdScript] : [tsxCli, qmdScript];

let projectDir: string;
let configDir: string;
let outsideDir: string;

function runQmd(
  args: string[],
  env: Record<string, string> = {},
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.execPath, [...runnerArgs, ...args], {
      cwd: projectDir,
      env: {
        ...process.env,
        QMD_CONFIG_DIR: configDir,
        PWD: projectDir,
        QMD_DOCTOR_DEVICE_PROBE: "0",
        ...env,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (d: Buffer) => { stdout += d.toString(); });
    proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });
    proc.on("error", reject);
    proc.on("close", (code) => resolve({ stdout, stderr, exitCode: code ?? 1 }));
  });
}

function writeLocalConfig(body: string): void {
  writeFileSync(join(projectDir, ".qmd", "index.yml"), body, "utf-8");
}

beforeEach(() => {
  projectDir = mkdtempSync(join(tmpdir(), "qmd-path-trust-proj-"));
  configDir = mkdtempSync(join(tmpdir(), "qmd-path-trust-cfg-"));
  outsideDir = mkdtempSync(join(tmpdir(), "qmd-path-trust-out-"));
  mkdirSync(join(projectDir, ".qmd"), { recursive: true });
  mkdirSync(join(projectDir, "docs"), { recursive: true });
  writeFileSync(join(projectDir, "docs", "readme.md"), "# Readme\n\nSome indexable content.\n", "utf-8");
  writeFileSync(join(outsideDir, "secret.md"), "# Secret\n\nShould not be indexed unattended.\n", "utf-8");
});

afterEach(() => {
  for (const dir of [projectDir, configDir, outsideDir]) {
    try { rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
  }
});

describe("qmd update with a checked-in collection path outside the project", () => {
  function outsideConfig(): string {
    return [
      "collections:",
      "  secrets:",
      `    path: ${JSON.stringify(outsideDir)}`,
      '    pattern: "**/*.md"',
      "",
    ].join("\n");
  }

  test("does not index the outside path unattended", async () => {
    writeLocalConfig(outsideConfig());
    const result = await runQmd(["update"]);

    expect(result.stdout).toContain("Collection paths outside this project");
    expect(result.stdout).toContain("qmd trust");
    expect(result.stdout).toContain(`Skipping collection 'secrets'`);
    expect(result.stdout).not.toContain("Indexed: 1 new");
    expect(result.exitCode).toBe(0);
  }, 120_000);

  test("indexes it after `qmd trust`", async () => {
    writeLocalConfig(outsideConfig());
    const trust = await runQmd(["trust"]);
    expect(trust.stdout).toContain("Trusted");
    expect(trust.stdout).toContain(outsideDir);

    const result = await runQmd(["update"]);
    expect(result.stdout).toContain("Indexed: 1 new");
    expect(result.stdout).not.toContain("Skipping collection 'secrets'");
  }, 120_000);

  test("re-arms the gate when the path is rewritten after approval", async () => {
    writeLocalConfig(outsideConfig());
    await runQmd(["trust"]);
    const other = mkdtempSync(join(tmpdir(), "qmd-path-trust-other-"));
    writeFileSync(join(other, "other.md"), "# Other\n\nAlso outside.\n", "utf-8");
    writeLocalConfig([
      "collections:",
      "  secrets:",
      `    path: ${JSON.stringify(other)}`,
      '    pattern: "**/*.md"',
      "",
    ].join("\n"));

    const result = await runQmd(["update"]);
    expect(result.stdout).toContain("Skipping collection 'secrets'");
    expect(result.stdout).not.toContain("Indexed: 1 new");
    try { rmSync(other, { recursive: true, force: true }); } catch { /* ignore */ }
  }, 120_000);

  test("QMD_TRUST_LOCAL_CONFIG=1 opts unattended runs back in", async () => {
    writeLocalConfig(outsideConfig());
    const result = await runQmd(["update"], { QMD_TRUST_LOCAL_CONFIG: "1" });
    expect(result.stdout).toContain("Indexed: 1 new");
    expect(result.stdout).not.toContain("Skipping collection 'secrets'");
  }, 120_000);
});

describe("qmd update with a checked-in custom model URI", () => {
  test("does not treat the custom model as trusted, but still indexes the project", async () => {
    writeLocalConfig([
      "collections:",
      "  docs:",
      "    path: ./docs",
      '    pattern: "**/*.md"',
      "models:",
      "  embed: hf:evil/embed/x.gguf",
      "",
    ].join("\n"));

    const result = await runQmd(["update"]);
    expect(result.stdout).toContain("Custom models");
    expect(result.stdout).toContain("hf:evil/embed/x.gguf");
    expect(result.stdout).toContain("Indexed: 1 new");
    expect(result.exitCode).toBe(0);
  }, 120_000);

  test("`qmd trust` records the custom model", async () => {
    writeLocalConfig([
      "collections:",
      "  docs:",
      "    path: ./docs",
      '    pattern: "**/*.md"',
      "models:",
      "  embed: hf:evil/embed/x.gguf",
      "",
    ].join("\n"));

    const trust = await runQmd(["trust"]);
    expect(trust.stdout).toContain("hf:evil/embed/x.gguf");
    expect(trust.stdout).toContain("Trusted");

    const result = await runQmd(["update"]);
    expect(result.stdout).not.toContain("Custom models");
    expect(result.stdout).toContain("Indexed: 1 new");
  }, 120_000);
});

describe("qmd update with only in-project paths", () => {
  test("indexes without a trust prompt", async () => {
    writeLocalConfig([
      "collections:",
      "  docs:",
      "    path: ./docs",
      '    pattern: "**/*.md"',
      "",
    ].join("\n"));

    const result = await runQmd(["update"]);
    expect(result.stdout).toContain("Indexed: 1 new");
    expect(result.stdout).not.toContain("qmd trust");
    expect(result.exitCode).toBe(0);
  }, 120_000);
});

describe("qmd update with a checked-in remote provider", () => {
  function providerConfig(baseUrl: string): string {
    return [
      "collections:",
      "  docs:",
      "    path: ./docs",
      '    pattern: "**/*.md"',
      "models:",
      "  provider:",
      `    base_url: ${JSON.stringify(baseUrl)}`,
      "    embed: cohere.embed-v4:0",
      "",
    ].join("\n");
  }

  test("gates the provider with an API key present and re-arms after an endpoint change", async () => {
    writeLocalConfig(providerConfig("https://one.example/models"));
    const untrusted = await runQmd(["update"], { QMD_API_KEY: "test-secret" });
    expect(untrusted.stdout).toContain("Remote provider:");
    expect(untrusted.stdout).toContain("qmd trust");
    expect(untrusted.stdout).toContain("Indexed: 1 new");

    const approved = await runQmd(["trust"]);
    expect(approved.stdout).toContain("https://one.example/models");
    writeLocalConfig(providerConfig("https://two.example/models"));
    const changed = await runQmd(["update"], { QMD_API_KEY: "test-secret" });
    expect(changed.stdout).toContain("Remote provider:");
    expect(changed.stdout).toContain("https://two.example/models");
  }, 120_000);
});
