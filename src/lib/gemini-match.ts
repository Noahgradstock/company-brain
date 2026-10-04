import type { Fact } from "@/trust-engine";

// Gemini only maps a free-text question to one known topic. It never scores: the trust engine
// stays deterministic. Returns null (keyword fallback) when no key is set or the call fails.
// Fastest first: routing to one of a few topics is an easy task.
// gemini-3.1-flash-lite took 5-16 s on the free tier and answered "none" to Dutch/French questions;
// 3.5-flash-lite answers correctly in ~2 s. 3.7-flash was dropped as fallback: it was overloaded (503).
const MODELS = ["gemini-3.5-flash-lite", "gemini-flash-lite-latest"];
const cache = new Map<string, string | null>();

export async function matchFactWithGemini(question: string, facts: Fact[]): Promise<string | null> {
  const key = process.env.GEMINI_API_KEY;
  if (!key) return null;
  const cacheKey = question.trim().toLowerCase();
  if (cache.has(cacheKey)) return cache.get(cacheKey)!;

  const keys = [...facts.map((f) => f.key), "none"];
  const system =
    "You route HR and payroll questions from SD Worx consultants to one topic in a knowledge base. " +
    'Answer with the single best topic key, or "none" if no topic fits. Questions may be in English, Dutch or French.';
  const input = `Topics:\n${facts.map((f) => `- ${f.key}: ${f.label}`).join("\n")}\n\nQuestion: "${question}"`;
  const schema = { type: "object", properties: { factKey: { type: "string", enum: keys } }, required: ["factKey"] };

  const call = (model: string) =>
    fetch("https://generativelanguage.googleapis.com/v1beta/interactions", {
      method: "POST",
      headers: { "content-type": "application/json", "x-goog-api-key": key },
      signal: AbortSignal.timeout(6_000),
      body: JSON.stringify({
        model,
        system_instruction: system,
        input,
        // Picking a topic needs no reasoning. Default thinking took 8-16 s on the free tier and
        // blew the 6 s timeout; minimal answers the same in 2-4 s.
        generation_config: { thinking_level: "minimal" },
        response_format: { type: "text", mime_type: "application/json", schema },
      }),
    });

  try {
    // Free tier is often overloaded (503/429), so fall through to other models.
    let res = await call(MODELS[0]);
    for (const model of MODELS.slice(1)) {
      if (res.status !== 503 && res.status !== 429) break;
      res = await call(model);
    }
    if (!res.ok) throw new Error(`Gemini ${res.status}`);
    const data = await res.json();
    const output = (data?.steps ?? []).findLast((s: { type?: string }) => s.type === "model_output");
    const text: string | undefined = output?.content?.find((c: { type?: string }) => c.type === "text")?.text;
    const factKey = text ? JSON.parse(text).factKey : null;
    const match = typeof factKey === "string" && facts.some((f) => f.key === factKey) ? factKey : null;
    if (cache.size >= 1_000) cache.delete(cache.keys().next().value!); // bounded: drop the oldest entry
    cache.set(cacheKey, match);
    return match;
  } catch (err) {
    console.error("Gemini topic match failed, using keywords:", err);
    return null;
  }
}
