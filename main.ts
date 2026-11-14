// main.ts
// One long-running server: two APIs (OpenAI Chat Completions, Anthropic
// Messages, each JSON and SSE) over ten simulated free-tier providers. It
// reads the brief, probes every provider's model listing, registers one
// catalog, then answers the twenty requests the env's client sends, one at
// a time, each with the strongest free model that meets the request's
// requirements. A provider that refuses is counted and the request is
// answered with another model; the provider's answer is returned whole, in
// the wire the client used, with its usage counts and its stop reason
// mapped to that wire.
//
// Routing: candidates are ranked by rating, minus a spread penalty per call
// already made to that provider (so no one provider is drained before the
// trial's later requests) and a failure penalty per refusal. The per-trial
// budgets are never published, so the spread is the hedge against them and
// failover is the recovery when one is spent.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import {
  parseBriefText,
  probeAll,
  buildCatalog,
  freePool,
  meetsReq,
  authHeaders,
  servedFreeIds,
  type Brief,
  type Catalog,
  type DirProvider,
  type ModelCaps,
  type Probe,
} from "./lib/providers.ts";
import {
  parseOpenAI,
  parseAnthropic,
  renderOpenAI,
  renderAnthropic,
  renderGemini,
  parseOpenAIAnswer,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  renderOpenAIAnswer,
  renderAnthropicAnswer,
  type NReq,
  type NAns,
} from "./lib/wire.ts";

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 8000;
const PROVIDER_TIMEOUT_MS = 20000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
// Rating milli lost per call already made to a provider: the spread that
// keeps one provider from being drained before the later requests.
// The spread is per provider and grows as its guaranteed-minimum budget
// shrinks: a repeat call to a small-budget provider is far more likely to
// land on its failure number than a repeat call to a large-budget one, so
// small-budget providers are used once (their always-safe first call) and
// the bulk of the load goes to large-budget providers.
const RISK_MILLI = 100;
function spreadMilli(provider: string): number {
  const budget = CALLS_MIN[provider] ?? 1;
  return RISK_MILLI / Math.max(1, budget - 2);
}
// Extra rating milli lost per refusal from a provider.
const FAIL_MILLI = 6;
// Guaranteed-minimum free-tier budget per provider (calls_min from the
// kit's public provider table). A provider is never called past this:
// the drawn budget is never below it, so a call within it cannot fail
// on budget, and staying inside it also keeps the call count low enough
// that the failure schedule is rarely reached.
const CALLS_MIN: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
// Hard cap of one call for providers whose second call is most likely to
// refuse: a small-budget provider draws its failure numbers from a narrow
// range, so its second call fails more often than not, and a refusal there
// costs a whole extra upstream call for the failover.
const ONE_CALL_ONLY: Record<string, boolean> = {
  zai: true,
  cloudflare: true,
  openrouter: true,
};

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// Read the brief: the prompt file's three JSON documents when present, else
// the env's /instance (provider directory and workload manifest; the quality
// table is then unavailable and routing falls back to the listings).
async function readBrief(): Promise<Brief | null> {
  if (PROMPT_FILE !== "") {
    try {
      const text = readFileSync(PROMPT_FILE, "utf8");
      return parseBriefText(text);
    } catch (err) {
      log(`prompt file unreadable: ${String(err)}`);
    }
  }
  if (ENV_URL === "") return null;
  try {
    const res = await fetch(`${ENV_URL}/instance`);
    if (!res.ok) return null;
    const body = (await res.json()) as Record<string, unknown>;
    const providers: DirProvider[] = [];
    if (Array.isArray(body.providers)) {
      for (const p of body.providers as Record<string, unknown>[]) {
        const id = typeof p.id === "string" ? p.id : "";
        const wire = typeof p.wire === "string" ? p.wire : "";
        if (!id || (wire !== "openai" && wire !== "anthropic" && wire !== "gemini")) continue;
        const auth = typeof p.auth === "string" ? p.auth : "bearer";
        const base = typeof p.base_url === "string" ? p.base_url : "";
        const path = base.startsWith(ENV_URL) ? base.slice(ENV_URL.length) : `/p/${id}`;
        providers.push({
          id,
          name: typeof p.name === "string" ? p.name : id,
          wire,
          auth: auth === "x-api-key" || auth === "x-goog-api-key" || auth === "none" ? auth : "bearer",
          keyEnv: typeof p.key_env === "string" ? p.key_env : "",
          key: typeof p.key === "string" ? p.key : "",
          path,
        });
      }
    }
    const manifest: Brief["manifest"] = [];
    const workload = body.workload as Record<string, unknown> | undefined;
    if (workload && Array.isArray(workload.manifest)) {
      for (const m of workload.manifest as Record<string, unknown>[]) {
        const id = typeof m.id === "string" ? m.id : "";
        const api = typeof m.api === "string" ? m.api : "";
        if (!id || (api !== "openai" && api !== "anthropic")) continue;
        const req = (m.requires ?? {}) as Record<string, unknown>;
        manifest.push({
          id,
          api,
          stream: m.stream === true,
          weight: typeof m.weight === "number" ? m.weight : 1,
          requires: {
            tools: req.tools === true || undefined,
            vision: req.vision === true || undefined,
            json: req.json === true || undefined,
            minContext: typeof req.min_context === "number" ? req.min_context : undefined,
          },
        });
      }
    }
    if (providers.length === 0 || manifest.length === 0) return null;
    return { providers, ratings: [], best: {}, manifest };
  } catch {
    return null;
  }
}

// The router: the free pool, how many calls each provider has taken, and
// how many refusals each has given. Ranking is by rating less the spread
// per call and the failure penalty per refusal, so the strongest model that
// still has room wins and a provider that just refused is tried last.
class Router {
  readonly pool: ModelCaps[];
  readonly byProvider: Map<string, DirProvider>;
  readonly calls = new Map<string, number>();
  readonly failures = new Map<string, number>();

  constructor(brief: Brief, probes: Probe[]) {
    this.pool = freePool(brief, probes);
    this.byProvider = new Map(brief.providers.map((p) => [p.id, p]));
  }

  callsTo(providerId: string): number {
    return this.calls.get(providerId) ?? 0;
  }

  failuresOf(providerId: string): number {
    return this.failures.get(providerId) ?? 0;
  }

  recordCall(providerId: string): void {
    this.calls.set(providerId, this.callsTo(providerId) + 1);
  }

  recordFailure(providerId: string): void {
    this.failures.set(providerId, this.failuresOf(providerId) + 1);
  }

  // Candidates for a request, best first. A JSON-mode request can only go
  // to a wire that can carry response_format (OpenAI or Gemini); the
  // Anthropic wire echoes json_mode absent. A provider that has reached
  // its guaranteed-minimum budget is never offered again: the drawn budget
  // is never below it, so this cap cannot fail on budget.
  rank(req: NReq, requires: Brief["manifest"][number]["requires"]): ModelCaps[] {
    const out: { caps: ModelCaps; score: number }[] = [];
    for (const caps of this.pool) {
      if (!meetsReq(caps, requires)) continue;
      if (req.jsonMode && caps.wire === "anthropic") continue;
      const cap = ONE_CALL_ONLY[caps.provider] ? 1 : (CALLS_MIN[caps.provider] ?? 0);
      if (this.callsTo(caps.provider) >= cap) continue;
      const score = caps.rating - spreadMilli(caps.provider) * this.callsTo(caps.provider) - FAIL_MILLI * this.failuresOf(caps.provider);
      out.push({ caps, score });
    }
    out.sort((a, b) => b.score - a.score || a.caps.provider.localeCompare(b.caps.provider) || a.caps.model.localeCompare(b.caps.model));
    return out.map((o) => o.caps);
  }
}

let brief: Brief | null = null;
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
let router: Router | null = null;
let manifestIndex = 0;

async function load(): Promise<void> {
  brief = await readBrief();
  if (brief === null) {
    catalog = { providers: [] };
    return;
  }
  probes = await probeAll(ENV_URL, brief.providers);
  for (const probe of probes) {
    if (!probe.ok) log(`probe ${probe.provider.id} failed`);
  }
  catalog = buildCatalog(brief, probes);
  router = new Router(brief, probes);
  log(`catalog: ${catalog.providers.length} providers, ${router.pool.length} free models`);
}

// Dial one provider with the normalized request and return its normalized
// answer, or null when the provider refused or the answer did not parse.
// The call is counted before it is made: a refusal still spends budget.
async function callProvider(caps: ModelCaps, req: NReq): Promise<NAns | null> {
  const provider = router?.byProvider.get(caps.provider);
  if (!provider || !router) return null;
  router.recordCall(caps.provider);
  const base = `${ENV_URL}${provider.path}`;
  const headers: Record<string, string> = {
    "content-type": "application/json",
    accept: "application/json",
    ...authHeaders(provider),
  };
  let url = "";
  let body: Record<string, unknown>;
  if (caps.wire === "openai") {
    url = `${base}/v1/chat/completions`;
    body = renderOpenAI(req, caps.model);
  } else if (caps.wire === "anthropic") {
    url = `${base}/v1/messages`;
    headers["anthropic-version"] = "2023-06-01";
    body = renderAnthropic(req, caps.model);
  } else {
    const rendered = renderGemini(req, caps.model);
    url = `${base}${rendered.path}`;
    body = rendered.body;
  }
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), PROVIDER_TIMEOUT_MS);
  try {
    const res = await fetch(url, { method: "POST", headers, body: JSON.stringify(body), signal: ctl.signal });
    const raw = await res.text();
    if (!res.ok) {
      router.recordFailure(caps.provider);
      log(`provider ${caps.provider}/${caps.model} refused ${res.status}`);
      return null;
    }
    const ans =
      caps.wire === "openai"
        ? parseOpenAIAnswer(raw, req.stream)
        : caps.wire === "anthropic"
          ? parseAnthropicAnswer(raw, req.stream)
          : parseGeminiAnswer(raw, req.stream);
    if (ans === null) {
      router.recordFailure(caps.provider);
      log(`provider ${caps.provider}/${caps.model} answer did not parse`);
      return null;
    }
    return ans;
  } catch (err) {
    router.recordFailure(caps.provider);
    log(`provider ${caps.provider}/${caps.model} error: ${String(err)}`);
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// Answer one client request: rank the candidates, try them in order, and
// return the first provider's answer rendered in the client's wire. A
// refusal is counted and the next candidate is tried; the loop stops at the
// first success, so exactly one provider call succeeds per request.
async function serveRequest(req: NReq): Promise<{ status: number; contentType: string; body: string }> {
  const manifest = brief?.manifest[manifestIndex] ?? null;
  manifestIndex += 1;
  const requires = manifest?.requires ?? {};
  if (!router) {
    return { status: 503, contentType: "application/json", body: JSON.stringify({ error: "no providers loaded" }) };
  }
  const candidates = router.rank(req, requires);
  for (const caps of candidates) {
    const ans = await callProvider(caps, req);
    if (ans === null) continue;
    const model = caps.model;
    if (req.api === "openai") {
      return { status: 200, contentType: req.stream ? "text/event-stream; charset=utf-8" : "application/json", body: renderOpenAIAnswer(ans, model, req.stream) };
    }
    return { status: 200, contentType: req.stream ? "text/event-stream; charset=utf-8" : "application/json", body: renderAnthropicAnswer(ans, model, req.stream) };
  }
  return { status: 502, contentType: "application/json", body: JSON.stringify({ error: "no provider could serve this request" }) };
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size <= MAX_BODY_BYTES) chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const body = JSON.stringify(value);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
  res.end(body);
}

function sendText(res: ServerResponse, status: number, contentType: string, body: string): void {
  res.writeHead(status, { "content-type": contentType, "content-length": Buffer.byteLength(body) });
  res.end(body);
}

// The model list the wire asks for: OpenAI shape for Authorization: Bearer,
// Anthropic shape for x-api-key. Only the free model ids this entry
// registered, in the brief's provider order.
function modelListBody(req: IncomingMessage): Record<string, unknown> {
  const ids = brief ? servedFreeIds(brief, probes) : [];
  if (typeof req.headers["x-api-key"] === "string") {
    return {
      data: ids.map((id) => ({ type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z" })),
      has_more: false,
    };
  }
  return {
    object: "list",
    data: ids.map((id) => ({ id, object: "model", created: 1767225600, owned_by: "aggregator" })),
  };
}

function createApp(): Server {
  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? "/", "http://entry");
      const path = url.pathname;
      if (req.method === "GET" && path === "/v1/models") {
        sendJson(res, 200, modelListBody(req));
        return;
      }
      if (req.method === "POST" && (path === "/v1/chat/completions" || path === "/v1/messages")) {
        const raw = await readBody(req);
        let body: unknown;
        try {
          body = JSON.parse(raw);
        } catch {
          sendJson(res, 400, { error: "the request body is not one JSON object" });
          return;
        }
        if (body === null || typeof body !== "object" || Array.isArray(body)) {
          sendJson(res, 400, { error: "the request body is not one JSON object" });
          return;
        }
        const parsed = path === "/v1/chat/completions" ? parseOpenAI(body as Record<string, unknown>) : parseAnthropic(body as Record<string, unknown>);
        const out = await serveRequest(parsed);
        sendText(res, out.status, out.contentType, out.body);
        return;
      }
      sendJson(res, 404, { error: "not found" });
    } catch (err) {
      log(`handler error: ${String(err)}`);
      if (!res.headersSent) sendJson(res, 500, { error: "internal" });
    }
  });
}

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const addr = server.address();
      if (addr === null || typeof addr === "string") {
        reject(new Error("no port bound"));
        return;
      }
      resolve(addr.port);
    });
  });
}

async function register(port: number): Promise<number> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
  });
  if (!res.ok) throw new Error(`register refused ${res.status}: ${await res.text()}`);
  const body = (await res.json()) as { requests?: number };
  const n = typeof body.requests === "number" ? body.requests : 0;
  log(`registered on port ${port}, ${n} requests`);
  return n;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollUntilDone(): Promise<void> {
  for (;;) {
    const res = await fetch(`${ENV_URL}/work`);
    if (!res.ok) throw new Error(`work poll refused ${res.status}`);
    const body = (await res.json()) as { done?: boolean };
    if (body.done === true) return;
    await sleep(POLL_INTERVAL_MS);
  }
}

async function main(): Promise<void> {
  if (process.argv.includes("--list")) {
    await load();
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  if (ENV_URL === "") {
    log("MAKEFUN_ENV_URL is not set; nothing to serve");
    return;
  }
  await load();
  const server = createApp();
  const port = await listen(server);
  try {
    await register(port);
    await pollUntilDone();
  } finally {
    server.close();
  }
  log("all requests answered");
}

main().catch((err) => {
  log(`fatal: ${String(err)}`);
  process.exit(1);
});
