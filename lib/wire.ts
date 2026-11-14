// lib/wire.ts
// The wire translations. A client request (OpenAI or Anthropic) is parsed
// into one normalized shape, rendered into any provider's wire (OpenAI,
// Anthropic, Gemini), and the provider's answer is parsed back into one
// normalized shape and rendered into the client's wire. The provider echoes
// what it parsed, so every render carries the request's fields untouched:
// system prompt, user text, images, tools, stop sequences, max_tokens and
// JSON mode. Only the last user message's text and the system text feed the
// echo, and text blocks inside one message join with "\n" on every wire, so
// renders join a message's text blocks into a single part with "\n" before
// handing them to Gemini.

export type ClientApi = "openai" | "anthropic";
export type ProviderWire = "openai" | "anthropic" | "gemini";

export type NPart =
  | { k: "text"; t: string }
  | { k: "img"; mime: string; data: string }
  | { k: "imgurl"; url: string }
  | { k: "call"; id: string; name: string; args: string }
  | { k: "result"; id: string; text: string; err: boolean };

export interface NMsg {
  role: "user" | "assistant";
  parts: NPart[];
}

export interface NTool {
  name: string;
  desc?: string;
  params?: unknown;
}

export interface NReq {
  api: ClientApi;
  stream: boolean;
  system: string | null;
  msgs: NMsg[];
  tools: NTool[] | null;
  toolChoice: unknown;
  maxTok: number | null;
  stops: string[] | null;
  temp: number | null;
  topP: number | null;
  jsonMode: boolean;
}

export interface NAns {
  text: string;
  input: number | null;
  output: number | null;
  stop: string;
}

function isObj(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asNum(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function asStr(v: unknown): string | null {
  return typeof v === "string" ? v : null;
}

// A stop value as the provider reads it: a string becomes one entry, an
// array of strings stays, anything else is no stops.
function stringList(v: unknown): string[] | null {
  if (typeof v === "string") return [v];
  if (Array.isArray(v)) {
    const out = v.filter((x): x is string => typeof x === "string");
    return out.length > 0 ? out : null;
  }
  return null;
}

// The text of a block list, the way the echo reads it: text blocks joined
// with a newline. (A plain string is itself.)
function blocksText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content) {
    if (isObj(b) && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

// The text of an answer's content blocks: joined with nothing.
function answerText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  let out = "";
  for (const b of content) {
    if (isObj(b) && b.type === "text" && typeof b.text === "string") out += b.text;
  }
  return out;
}

// Split a data: URL into its mime type and base64 payload, or null.
function dataUrlParts(url: string): { mime: string; data: string } | null {
  if (!url.startsWith("data:")) return null;
  const comma = url.indexOf(",");
  if (comma < 0) return null;
  const head = url.slice(5, comma);
  const semi = head.indexOf(";");
  const mime = semi >= 0 ? head.slice(0, semi) : head;
  if (!head.includes("base64")) return null;
  return { mime: mime || "application/octet-stream", data: url.slice(comma + 1) };
}

// Split a server-sent-events body into its frames.
export interface SseFrame {
  event: string | null;
  data: string;
}

export function parseSse(raw: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const chunks = raw.split(/\r?\n\r?\n/);
  for (const chunk of chunks) {
    let event: string | null = null;
    let data = "";
    let saw = false;
    for (const line of chunk.split(/\r?\n/)) {
      if (line.startsWith("event:")) {
        event = line.slice("event:".length).trim();
        saw = true;
      } else if (line.startsWith("data:")) {
        data += (data ? "\n" : "") + line.slice("data:".length).trimStart();
        saw = true;
      }
    }
    if (saw) frames.push({ event, data });
  }
  return frames;
}

// The stop-reason family a provider's raw stop belongs to: 0 stop, 1 length,
// 2 tool call, 3 refusal. The provider only ever answers family 0, but the
// mapping stays total so a surprise still lands in the right family.
function stopFamily(stop: string): number {
  const s = stop.toLowerCase();
  if (s === "stop" || s === "end_turn" || s === "stop_sequence") return 0;
  if (s === "length" || s === "max_tokens") return 1;
  if (s === "tool_calls" || s === "tool_use" || s === "function_call") return 2;
  if (s === "content_filter" || s === "refusal") return 3;
  return 0;
}

export function stopForWire(stop: string, wire: ProviderWire | ClientApi): string {
  const fam = stopFamily(stop);
  if (wire === "openai") return ["stop", "length", "tool_calls", "content_filter"][fam];
  if (wire === "anthropic") return ["end_turn", "max_tokens", "tool_use", "refusal"][fam];
  return ["STOP", "MAX_TOKENS", "TOOL_CALLS", "SAFETY"][fam];
}

// Parse an OpenAI Chat Completions request body into the normalized shape.
export function parseOpenAI(body: Record<string, unknown>): NReq {
  const systemParts: string[] = [];
  const msgs: NMsg[] = [];
  const raw = Array.isArray(body.messages) ? body.messages.filter(isObj) : [];
  for (const m of raw) {
    const role = asStr(m.role);
    if (role === "system") {
      const t = blocksText(m.content);
      if (t !== "") systemParts.push(t);
      continue;
    }
    if (role === "user") {
      const parts: NPart[] = [];
      if (Array.isArray(m.content)) {
        for (const b of m.content) {
          if (!isObj(b)) continue;
          if (b.type === "text" && typeof b.text === "string") parts.push({ k: "text", t: b.text });
          else if (b.type === "image_url" && isObj(b.image_url) && typeof b.image_url.url === "string") {
            const d = dataUrlParts(b.image_url.url);
            if (d) parts.push({ k: "img", mime: d.mime, data: d.data });
            else parts.push({ k: "imgurl", url: b.image_url.url });
          }
        }
      } else if (typeof m.content === "string" && m.content !== "") {
        parts.push({ k: "text", t: m.content });
      }
      msgs.push({ role: "user", parts });
      continue;
    }
    if (role === "assistant") {
      const parts: NPart[] = [];
      const t = blocksText(m.content);
      if (t !== "") parts.push({ k: "text", t });
      if (Array.isArray(m.tool_calls)) {
        for (const c of m.tool_calls) {
          if (!isObj(c) || !isObj(c.function)) continue;
          const name = asStr(c.function.name);
          if (!name) continue;
          parts.push({
            k: "call",
            id: asStr(c.id) ?? "",
            name,
            args: typeof c.function.arguments === "string" ? c.function.arguments : "{}",
          });
        }
      }
      msgs.push({ role: "assistant", parts });
      continue;
    }
    if (role === "tool") {
      const t = blocksText(m.content);
      msgs.push({
        role: "user",
        parts: [{ k: "result", id: asStr(m.tool_call_id) ?? "", text: t, err: false }],
      });
      continue;
    }
  }
  const tools: NTool[] | null = Array.isArray(body.tools)
    ? body.tools
        .filter(isObj)
        .map((t) => {
          const fn = isObj(t.function) ? t.function : {};
          return {
            name: asStr(fn.name) ?? "",
            desc: asStr(fn.description) ?? undefined,
            params: fn.parameters,
          };
        })
        .filter((t) => t.name !== "")
    : null;
  const fmt = isObj(body.response_format) ? body.response_format.type : undefined;
  return {
    api: "openai",
    stream: body.stream === true,
    system: systemParts.length > 0 ? systemParts.join("\n") : null,
    msgs,
    tools: tools && tools.length > 0 ? tools : null,
    toolChoice: body.tool_choice ?? null,
    maxTok: asNum(body.max_tokens) ?? asNum(body.max_completion_tokens),
    stops: stringList(body.stop),
    temp: asNum(body.temperature),
    topP: asNum(body.top_p),
    jsonMode: fmt === "json_object" || fmt === "json_schema",
  };
}

// Parse an Anthropic Messages request body into the normalized shape.
export function parseAnthropic(body: Record<string, unknown>): NReq {
  const sysText = blocksText(body.system);
  const msgs: NMsg[] = [];
  const raw = Array.isArray(body.messages) ? body.messages.filter(isObj) : [];
  for (const m of raw) {
    const role = asStr(m.role);
    if (role !== "user" && role !== "assistant") continue;
    const parts: NPart[] = [];
    if (typeof m.content === "string" && m.content !== "") {
      parts.push({ k: "text", t: m.content });
    } else if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (!isObj(b)) continue;
        if (b.type === "text" && typeof b.text === "string") parts.push({ k: "text", t: b.text });
        else if (b.type === "image" && isObj(b.source)) {
          if (b.source.type === "base64" && typeof b.source.data === "string") {
            parts.push({ k: "img", mime: asStr(b.source.media_type) ?? "image/jpeg", data: b.source.data });
          } else if (b.source.type === "url" && typeof b.source.url === "string") {
            parts.push({ k: "imgurl", url: b.source.url });
          }
        } else if (b.type === "tool_use" && typeof b.name === "string") {
          parts.push({
            k: "call",
            id: asStr(b.id) ?? "",
            name: b.name,
            args: b.input !== undefined ? JSON.stringify(b.input) : "{}",
          });
        } else if (b.type === "tool_result") {
          const content = b.content;
          const t = typeof content === "string" ? content : blocksText(content);
          parts.push({ k: "result", id: asStr(b.tool_use_id) ?? "", text: t, err: b.is_error === true });
        }
      }
    }
    msgs.push({ role, parts });
  }
  const tools: NTool[] | null = Array.isArray(body.tools)
    ? body.tools
        .filter(isObj)
        .map((t) => ({
          name: asStr(t.name) ?? "",
          desc: asStr(t.description) ?? undefined,
          params: t.input_schema,
        }))
        .filter((t) => t.name !== "")
    : null;
  return {
    api: "anthropic",
    stream: body.stream === true,
    system: sysText !== "" ? sysText : null,
    msgs,
    tools: tools && tools.length > 0 ? tools : null,
    toolChoice: body.tool_choice ?? null,
    maxTok: asNum(body.max_tokens),
    stops: stringList(body.stop_sequences),
    temp: asNum(body.temperature),
    topP: asNum(body.top_p),
    jsonMode: false,
  };
}

// Map a tool choice into the OpenAI wire's shape.
function toolChoiceOpenAI(tc: unknown): unknown {
  if (tc === "auto" || tc === "none" || tc === "required") return tc;
  if (isObj(tc)) {
    if (tc.type === "auto") return "auto";
    if (tc.type === "any") return "required";
    if (tc.type === "tool" && typeof tc.name === "string") {
      return { type: "function", function: { name: tc.name } };
    }
    if (tc.type === "function" && isObj(tc.function) && typeof tc.function.name === "string") return tc;
  }
  return undefined;
}

// Map a tool choice into the Anthropic wire's shape.
function toolChoiceAnthropic(tc: unknown): unknown {
  if (isObj(tc)) {
    if (tc.type === "auto" || tc.type === "any" || tc.type === "tool") return tc;
    if (tc.type === "function" && isObj(tc.function) && typeof tc.function.name === "string") {
      return { type: "tool", name: tc.function.name };
    }
  }
  if (tc === "auto") return { type: "auto" };
  if (tc === "required") return { type: "any" };
  return undefined;
}

// Render the normalized request as an OpenAI Chat Completions body.
// Only fields the client sent are set: a default max_tokens or stop list
// would echo a value the client never sent and lose the request.
export function renderOpenAI(req: NReq, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  if (req.system !== null) messages.push({ role: "system", content: req.system });
  for (const msg of req.msgs) {
    if (msg.role === "user") {
      const content: Record<string, unknown>[] = [];
      const flush = () => {
        if (content.length > 0) {
          // Copy: the array is cleared after the push, so the pushed
          // message must not share it.
          messages.push({ role: "user", content: [...content] });
          content.length = 0;
        }
      };
      for (const p of msg.parts) {
        if (p.k === "text") content.push({ type: "text", text: p.t });
        else if (p.k === "img") content.push({ type: "image_url", image_url: { url: `data:${p.mime};base64,${p.data}` } });
        else if (p.k === "imgurl") content.push({ type: "image_url", image_url: { url: p.url } });
        else if (p.k === "result") {
          flush();
          messages.push({ role: "tool", tool_call_id: p.id, content: p.text });
        }
      }
      flush();
    } else {
      const text = msg.parts.filter((p) => p.k === "text").map((p) => (p.k === "text" ? p.t : "")).join("");
      const calls = msg.parts.filter((p) => p.k === "call");
      const m: Record<string, unknown> = { role: "assistant", content: text };
      if (calls.length > 0) {
        m.tool_calls = calls.map((p) =>
          p.k === "call"
            ? { id: p.id, type: "function", function: { name: p.name, arguments: p.args } }
            : null,
        );
      }
      messages.push(m);
    }
  }
  const body: Record<string, unknown> = { model, messages, stream: req.stream };
  if (req.tools) {
    body.tools = req.tools.map((t) => ({
      type: "function",
      function: { name: t.name, ...(t.desc !== undefined ? { description: t.desc } : {}), ...(t.params !== undefined ? { parameters: t.params } : {}) },
    }));
  }
  const tc = toolChoiceOpenAI(req.toolChoice);
  if (tc !== undefined) body.tool_choice = tc;
  if (req.maxTok !== null) body.max_tokens = req.maxTok;
  if (req.stops !== null) body.stop = req.stops;
  if (req.temp !== null) body.temperature = req.temp;
  if (req.topP !== null) body.top_p = req.topP;
  if (req.jsonMode) body.response_format = { type: "json_object" };
  if (req.stream) body.stream_options = { include_usage: true };
  return body;
}

// Render the normalized request as an Anthropic Messages body. The Anthropic
// wire cannot carry JSON mode, so callers must not route a JSON-mode request
// here; max_tokens is only set when the client sent one.
export function renderAnthropic(req: NReq, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  for (const msg of req.msgs) {
    const content: Record<string, unknown>[] = [];
    for (const p of msg.parts) {
      if (p.k === "text") content.push({ type: "text", text: p.t });
      else if (p.k === "img") {
        content.push({ type: "image", source: { type: "base64", media_type: p.mime, data: p.data } });
      } else if (p.k === "imgurl") {
        content.push({ type: "image", source: { type: "url", url: p.url } });
      } else if (p.k === "call") {
        let input: unknown = {};
        try {
          input = JSON.parse(p.args);
        } catch {
          input = {};
        }
        content.push({ type: "tool_use", id: p.id, name: p.name, input });
      } else if (p.k === "result") {
        const block: Record<string, unknown> = { type: "tool_result", tool_use_id: p.id, content: p.text };
        if (p.err) block.is_error = true;
        content.push(block);
      }
    }
    messages.push({ role: msg.role, content });
  }
  const body: Record<string, unknown> = { model, messages, stream: req.stream };
  if (req.system !== null) body.system = req.system;
  if (req.tools) {
    body.tools = req.tools.map((t) => ({
      name: t.name,
      ...(t.desc !== undefined ? { description: t.desc } : {}),
      ...(t.params !== undefined ? { input_schema: t.params } : {}),
    }));
  }
  const tc = toolChoiceAnthropic(req.toolChoice);
  if (tc !== undefined) body.tool_choice = tc;
  if (req.maxTok !== null) body.max_tokens = req.maxTok;
  if (req.stops !== null) body.stop_sequences = req.stops;
  if (req.temp !== null) body.temperature = req.temp;
  if (req.topP !== null) body.top_p = req.topP;
  return body;
}

// Render the normalized request as a Gemini generateContent body, plus the
// URL path that names the model and the stream. A message's text blocks are
// joined into one part with "\n": Gemini joins a content's text parts with
// "\n" for the echo, the same join the OpenAI and Anthropic wires use.
export function renderGemini(req: NReq, model: string): { path: string; body: Record<string, unknown> } {
  const contents: Record<string, unknown>[] = [];
  for (const msg of req.msgs) {
    const parts: Record<string, unknown>[] = [];
    // One part: Gemini joins a content's text parts with "\n" for the echo,
    // while the OpenAI and Anthropic wires join them with "\n" too, so the
    // join here must be "\n" to reproduce the same user text.
    const text = msg.parts.filter((p) => p.k === "text").map((p) => (p.k === "text" ? p.t : "")).join("\n");
    if (text !== "") parts.push({ text });
    for (const p of msg.parts) {
      if (p.k === "img") parts.push({ inlineData: { mimeType: p.mime, data: p.data } });
      else if (p.k === "imgurl") parts.push({ fileData: { fileUri: p.url } });
      else if (p.k === "call") {
        let args: unknown = {};
        try {
          args = JSON.parse(p.args);
        } catch {
          args = {};
        }
        parts.push({ functionCall: { name: p.name, args } });
      } else if (p.k === "result") {
        parts.push({ functionResponse: { name: p.id, response: { result: p.text } } });
      }
    }
    contents.push({ role: msg.role === "assistant" ? "model" : "user", parts });
  }
  const body: Record<string, unknown> = { contents };
  if (req.system !== null) body.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools) {
    body.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({
          name: t.name,
          ...(t.desc !== undefined ? { description: t.desc } : {}),
          ...(t.params !== undefined ? { parameters: t.params } : {}),
        })),
      },
    ];
  }
  const config: Record<string, unknown> = {};
  if (req.maxTok !== null) config.maxOutputTokens = req.maxTok;
  if (req.stops !== null) config.stopSequences = req.stops;
  if (req.temp !== null) config.temperature = req.temp;
  if (req.topP !== null) config.topP = req.topP;
  if (req.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  const verb = req.stream ? "streamGenerateContent" : "generateContent";
  return { path: `/v1beta/models/${encodeURIComponent(model)}:${verb}`, body };
}

// Parse a provider's OpenAI-wire answer (JSON, or SSE when streamed) into
// the normalized shape. The text is carried whole; usage and stop come from
// the provider's own fields.
export function parseOpenAIAnswer(raw: string, stream: boolean): NAns | null {
  if (stream) {
    let text = "";
    let input: number | null = null;
    let output: number | null = null;
    let stop: string | null = null;
    for (const frame of parseSse(raw)) {
      if (frame.data === "[DONE]") continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isObj(chunk)) continue;
      if (isObj(chunk.usage)) {
        input = asNum(chunk.usage.prompt_tokens) ?? input;
        output = asNum(chunk.usage.completion_tokens) ?? output;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const first = isObj(choices[0]) ? choices[0] : null;
      if (!first) continue;
      if (isObj(first.delta) && typeof first.delta.content === "string") text += first.delta.content;
      if (typeof first.finish_reason === "string") stop = first.finish_reason;
    }
    if (stop === null) return null;
    return { text, input, output, stop };
  }
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isObj(value)) return null;
  const choices = Array.isArray(value.choices) ? value.choices : [];
  const first = isObj(choices[0]) ? choices[0] : null;
  if (!first || !isObj(first.message)) return null;
  const stop = asStr(first.finish_reason);
  if (stop === null) return null;
  const text = typeof first.message.content === "string" ? first.message.content : answerText(first.message.content);
  const usage = isObj(value.usage) ? value.usage : null;
  return {
    text,
    input: usage ? asNum(usage.prompt_tokens) : null,
    output: usage ? asNum(usage.completion_tokens) : null,
    stop,
  };
}

// Parse a provider's Anthropic-wire answer into the normalized shape.
export function parseAnthropicAnswer(raw: string, stream: boolean): NAns | null {
  if (stream) {
    let text = "";
    let input: number | null = null;
    let output: number | null = null;
    let stop: string | null = null;
    for (const frame of parseSse(raw)) {
      let data: unknown;
      try {
        data = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isObj(data)) continue;
      if (frame.event === "message_start" && isObj(data.message) && isObj(data.message.usage)) {
        input = asNum(data.message.usage.input_tokens) ?? input;
      }
      if (frame.event === "content_block_delta" && isObj(data.delta) && data.delta.type === "text_delta" && typeof data.delta.text === "string") {
        text += data.delta.text;
      }
      if (frame.event === "message_delta") {
        if (isObj(data.delta) && typeof data.delta.stop_reason === "string") stop = data.delta.stop_reason;
        if (isObj(data.usage)) output = asNum(data.usage.output_tokens) ?? output;
      }
    }
    if (stop === null) return null;
    return { text, input, output, stop };
  }
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isObj(value) || value.type !== "message" || !Array.isArray(value.content)) return null;
  const stop = asStr(value.stop_reason);
  if (stop === null) return null;
  const usage = isObj(value.usage) ? value.usage : null;
  return {
    text: answerText(value.content),
    input: usage ? asNum(usage.input_tokens) : null,
    output: usage ? asNum(usage.output_tokens) : null,
    stop,
  };
}

// The text of one Gemini candidate: its content parts' text, joined.
function geminiCandidateText(candidate: unknown): string {
  if (!isObj(candidate) || !isObj(candidate.content) || !Array.isArray(candidate.content.parts)) return "";
  let out = "";
  for (const part of candidate.content.parts) {
    if (isObj(part) && typeof part.text === "string") out += part.text;
  }
  return out;
}

// Parse a provider's Gemini-wire answer into the normalized shape.
export function parseGeminiAnswer(raw: string, stream: boolean): NAns | null {
  if (stream) {
    let text = "";
    let input: number | null = null;
    let output: number | null = null;
    let stop: string | null = null;
    for (const frame of parseSse(raw)) {
      let value: unknown;
      try {
        value = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isObj(value)) continue;
      const candidates = Array.isArray(value.candidates) ? value.candidates : [];
      const first = isObj(candidates[0]) ? candidates[0] : null;
      if (first) {
        text += geminiCandidateText(first);
        if (typeof first.finishReason === "string") stop = first.finishReason;
      }
      if (isObj(value.usageMetadata)) {
        input = asNum(value.usageMetadata.promptTokenCount) ?? input;
        output = asNum(value.usageMetadata.candidatesTokenCount) ?? output;
      }
    }
    if (stop === null) return null;
    return { text, input, output, stop };
  }
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isObj(value)) return null;
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const first = isObj(candidates[0]) ? candidates[0] : null;
  if (!first) return null;
  const stop = asStr(first.finishReason);
  if (stop === null) return null;
  const usage = isObj(value.usageMetadata) ? value.usageMetadata : null;
  return {
    text: geminiCandidateText(first),
    input: usage ? asNum(usage.promptTokenCount) : null,
    output: usage ? asNum(usage.candidatesTokenCount) : null,
    stop,
  };
}

// A fixed creation time and id keep every answer a pure function of the
// request and the provider's answer, so a replay exports byte-identical.
const CREATED = 1767225600;
const OPENAI_ID = "chatcmpl-agg";
const ANTHROPIC_ID = "msg_agg";

function splitPieces(text: string, size = 24): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out.length === 0 ? [""] : out;
}

// Render the normalized answer on the OpenAI wire, JSON or SSE. Streaming
// re-emits the buffered answer in order: role, content deltas, finish,
// usage, [DONE].
export function renderOpenAIAnswer(ans: NAns, model: string, stream: boolean): string {
  const input = ans.input ?? 0;
  const output = ans.output ?? 0;
  const stop = stopForWire(ans.stop, "openai");
  if (!stream) {
    return JSON.stringify({
      id: OPENAI_ID,
      object: "chat.completion",
      created: CREATED,
      model,
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: ans.text },
          finish_reason: stop,
          logprobs: null,
        },
      ],
      usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
    });
  }
  const head = { id: OPENAI_ID, object: "chat.completion.chunk", created: CREATED, model };
  let out = "";
  const frame = (obj: Record<string, unknown>) => {
    out += `data: ${JSON.stringify(obj)}\n\n`;
  };
  frame({ ...head, choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] });
  for (const piece of splitPieces(ans.text)) {
    frame({ ...head, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  frame({ ...head, choices: [{ index: 0, delta: {}, finish_reason: stop }] });
  frame({ ...head, choices: [], usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } });
  out += "data: [DONE]\n\n";
  return out;
}

// Render the normalized answer on the Anthropic wire, JSON or SSE. Streaming
// re-emits the buffered answer in order: message_start, content_block_start,
// text deltas, content_block_stop, message_delta, message_stop.
export function renderAnthropicAnswer(ans: NAns, model: string, stream: boolean): string {
  const input = ans.input ?? 0;
  const output = ans.output ?? 0;
  const stop = stopForWire(ans.stop, "anthropic");
  if (!stream) {
    return JSON.stringify({
      id: ANTHROPIC_ID,
      type: "message",
      role: "assistant",
      model,
      content: [{ type: "text", text: ans.text }],
      stop_reason: stop,
      stop_sequence: null,
      usage: { input_tokens: input, output_tokens: output },
    });
  }
  let out = "";
  const event = (name: string, data: Record<string, unknown>) => {
    out += `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
  };
  event("message_start", {
    type: "message_start",
    message: {
      id: ANTHROPIC_ID,
      type: "message",
      role: "assistant",
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: input, output_tokens: 0 },
    },
  });
  event("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
  for (const piece of splitPieces(ans.text)) {
    event("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } });
  }
  event("content_block_stop", { type: "content_block_stop", index: 0 });
  event("message_delta", {
    type: "message_delta",
    delta: { stop_reason: stop, stop_sequence: null },
    usage: { output_tokens: output },
  });
  event("message_stop", { type: "message_stop" });
  return out;
}
