// lib/providers.ts
// The brief's documents, the provider directory, the catalog probe, and the
// capability and rating tables this entry routes on. Nothing here dials a
// model: the probe only reads each provider's model listing, and routing
// reads the brief's quality table plus the manifest's requirements.

export type Wire = "openai" | "anthropic" | "gemini";
export type AuthKind = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

export interface DirProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthKind;
  keyEnv: string;
  key: string;
  path: string;
}

export interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  context: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface ManifestReq {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: {
    tools?: boolean;
    vision?: boolean;
    json?: boolean;
    minContext?: number;
  };
}

export interface Brief {
  providers: DirProvider[];
  ratings: RatingRow[];
  best: Record<string, number>;
  manifest: ManifestReq[];
}

export interface ListedModel {
  id: string;
  free: boolean;
  context: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

export interface Probe {
  provider: DirProvider;
  ok: boolean;
  models: ListedModel[];
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

function isObj(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asBool(v: unknown): boolean {
  return v === true;
}

function asNum(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function asStr(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

// Extract one balanced JSON object out of a longer text, starting at `at`.
// Returns the parsed value and the index just past its closing brace.
function balancedJson(text: string, at: number): { value: unknown; end: number } | null {
  let i = at;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (; i < text.length; i++) {
    const c = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === "{") depth++;
    else if (c === "}") {
      depth--;
      if (depth === 0) {
        try {
          return { value: JSON.parse(text.slice(at, i + 1)), end: i + 1 };
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

// Find the JSON document in `text` whose schema field equals `schema`.
function findDoc(text: string, schema: string): Record<string, unknown> | null {
  let from = 0;
  while (true) {
    const at = text.indexOf("{", from);
    if (at < 0) return null;
    const got = balancedJson(text, at);
    if (got === null) return null;
    if (isObj(got.value) && got.value.schema === schema) return got.value;
    from = got.end;
  }
}

const PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1";
const RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1";
const WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1";

// Parse the three JSON documents the brief carries: the provider directory,
// the quality table with each request's best rating, and the workload
// manifest. Throws when any of the three is missing.
export function parseBriefText(text: string): Brief {
  const provDoc = findDoc(text, PROVIDERS_SCHEMA);
  const rateDoc = findDoc(text, RATINGS_SCHEMA);
  const workDoc = findDoc(text, WORKLOAD_SCHEMA);
  if (!provDoc || !rateDoc || !workDoc) {
    throw new Error("brief is missing one of its three JSON documents");
  }
  const providers: DirProvider[] = [];
  if (Array.isArray(provDoc.providers)) {
    for (const p of provDoc.providers) {
      if (!isObj(p)) continue;
      const id = asStr(p.id);
      const wire = asStr(p.wire);
      if (!id || (wire !== "openai" && wire !== "anthropic" && wire !== "gemini")) continue;
      const auth = asStr(p.auth);
      providers.push({
        id,
        name: asStr(p.name) ?? id,
        wire,
        auth: auth === "x-api-key" || auth === "x-goog-api-key" || auth === "none" ? auth : "bearer",
        keyEnv: asStr(p.key_env) ?? "",
        key: asStr(p.key) ?? "",
        path: asStr(p.path) ?? `/p/${id}`,
      });
    }
  }
  const ratings: RatingRow[] = [];
  if (Array.isArray(rateDoc.ratings)) {
    for (const r of rateDoc.ratings) {
      if (!isObj(r)) continue;
      const model = asStr(r.model);
      const provider = asStr(r.provider);
      const rating = asNum(r.rating_milli);
      if (!model || !provider || rating === null) continue;
      ratings.push({
        model,
        provider,
        rating,
        context: asNum(r.context_length) ?? 0,
        tools: asBool(r.tools),
        vision: asBool(r.vision),
        json: asBool(r.json),
        free: asBool(r.free),
      });
    }
  }
  const best: Record<string, number> = {};
  if (Array.isArray(rateDoc.best)) {
    for (const b of rateDoc.best) {
      if (!isObj(b)) continue;
      const id = asStr(b.id);
      const rating = asNum(b.rating_milli);
      if (id && rating !== null) best[id] = rating;
    }
  }
  const manifest: ManifestReq[] = [];
  if (Array.isArray(workDoc.manifest)) {
    for (const m of workDoc.manifest) {
      if (!isObj(m)) continue;
      const id = asStr(m.id);
      const api = asStr(m.api);
      if (!id || (api !== "openai" && api !== "anthropic")) continue;
      const req = isObj(m.requires) ? m.requires : {};
      manifest.push({
        id,
        api,
        stream: m.stream === true,
        weight: asNum(m.weight) ?? 1,
        requires: {
          tools: asBool(req.tools) || undefined,
          vision: asBool(req.vision) || undefined,
          json: asBool(req.json) || undefined,
          minContext: asNum(req.min_context) ?? undefined,
        },
      });
    }
  }
  if (providers.length === 0 || manifest.length === 0) {
    throw new Error("brief documents parsed but hold no providers or requests");
  }
  return { providers, ratings, best, manifest };
}

// The auth header a provider's wire expects, or nothing when it takes none.
export function authHeaders(p: DirProvider): Record<string, string> {
  if (p.auth === "bearer") return { authorization: `Bearer ${p.key}` };
  if (p.auth === "x-api-key") return { "x-api-key": p.key };
  if (p.auth === "x-goog-api-key") return { "x-goog-api-key": p.key };
  return {};
}

// Read one provider's model listing in its own wire. The listing is the
// authority for what the provider serves: id, free flag, context length and
// the tools/vision/json capabilities. A probe that fails (or times out)
// leaves the provider out of the catalog rather than guessing.
export async function probeProvider(envUrl: string, p: DirProvider, timeoutMs = 8000): Promise<Probe> {
  const base = `${envUrl}${p.path}`;
  const url =
    p.wire === "gemini" ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: "application/json", ...authHeaders(p) };
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(url, { method: "GET", headers, signal: ctl.signal });
    if (!res.ok) return { provider: p, ok: false, models: [] };
    const body: unknown = await res.json();
    if (!isObj(body)) return { provider: p, ok: false, models: [] };
    const rows: unknown[] = p.wire === "gemini" ? (Array.isArray(body.models) ? body.models : []) : Array.isArray(body.data) ? body.data : [];
    const models: ListedModel[] = [];
    const seen = new Set<string>();
    for (const row of rows) {
      if (!isObj(row)) continue;
      let id = asStr(row.id);
      if (p.wire === "gemini") {
        const name = asStr(row.name);
        if (name && name.startsWith("models/")) id = name.slice("models/".length);
      }
      if (!id || seen.has(id)) continue;
      seen.add(id);
      const ctx = asNum(row.context_length) ?? asNum(row.inputTokenLimit) ?? 0;
      models.push({
        id,
        free: asBool(row.free),
        context: ctx,
        tools: asBool(row.tools),
        vision: asBool(row.vision),
        json: asBool(row.json),
      });
    }
    return { provider: p, ok: true, models };
  } catch {
    return { provider: p, ok: false, models: [] };
  } finally {
    clearTimeout(timer);
  }
}

// Probe every provider in the brief's order, one at a time (a provider
// serves one call at a time, and the probe is not the place to test that).
export async function probeAll(envUrl: string, providers: DirProvider[]): Promise<Probe[]> {
  const out: Probe[] = [];
  for (const p of providers) out.push(await probeProvider(envUrl, p));
  return out;
}

// The quality table's row for one (provider, model), or null when the table
// does not name it.
export function ratingFor(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  for (const r of brief.ratings) {
    if (r.provider === providerId && r.model === modelId) return r;
  }
  return null;
}

// Capabilities a model can actually serve: the listing's flags, with the
// quality table's row filling in where the listing is silent (a listing
// that hides a capability leaves the row short, so the table wins when it
// names the model).
export interface ModelCaps {
  provider: string;
  model: string;
  wire: Wire;
  free: boolean;
  context: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  rating: number;
}

export function capsOf(brief: Brief, probe: Probe, listed: ListedModel): ModelCaps {
  const row = ratingFor(brief, probe.provider.id, listed.id);
  const free = row ? row.free : listed.free;
  return {
    provider: probe.provider.id,
    model: listed.id,
    wire: probe.provider.wire,
    free,
    context: row?.context ?? listed.context,
    tools: row ? row.tools : listed.tools,
    vision: row ? row.vision : listed.vision,
    json: row ? row.json : listed.json,
    rating: row ? row.rating : 0,
  };
}

// Every free model every probed provider serves, in the brief's provider
// order. This is the pool routing draws from.
export function freePool(brief: Brief, probes: Probe[]): ModelCaps[] {
  const pool: ModelCaps[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const m of probe.models) {
      const caps = capsOf(brief, probe, m);
      if (caps.free) pool.push(caps);
    }
  }
  return pool;
}

// Whether a model meets one request's requirements.
export function meetsReq(caps: ModelCaps, req: ManifestReq["requires"]): boolean {
  if (req.tools && !caps.tools) return false;
  if (req.vision && !caps.vision) return false;
  if (req.json && !caps.json) return false;
  if (req.minContext !== undefined && caps.context < req.minContext) return false;
  return true;
}

// The catalog this entry registers and prints with --list: one entry per
// provider that answered its probe, models in the listing's own order, the
// free flag from the quality table where it names the model.
export function buildCatalog(brief: Brief, probes: Probe[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const models: CatalogModel[] = [];
    const seen = new Set<string>();
    for (const listed of probe.models) {
      if (seen.has(listed.id)) continue;
      seen.add(listed.id);
      const caps = capsOf(brief, probe, listed);
      models.push({ id: listed.id, free: caps.free, context_length: caps.context || null });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.path,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}

// The free model ids per provider, for GET /v1/models: the ids this entry
// registered, in the brief's provider order.
export function servedFreeIds(brief: Brief, probes: Probe[]): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const caps of freePool(brief, probes)) {
    if (!seen.has(caps.model)) {
      seen.add(caps.model);
      ids.push(caps.model);
    }
  }
  return ids;
}
