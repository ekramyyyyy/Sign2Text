// Cloudflare Pages Function: POST /api/translate
const SYSTEM =
  "You convert recognized American Sign Language (ASL) glosses into natural sentences. " +
  "Each numbered line is ONE sign with the recognizer's top candidates and confidences; pick the most plausible " +
  "candidate per position using sentence context. ASL gloss order and grammar differ from English " +
  "(e.g. 'YOU NAME WHAT' means 'What is your name?'). Add only the function words needed; do not invent content. " +
  'Reply with ONLY JSON: {"glosses": [chosen glosses], "english": "...", "arabic": "..."}. ' +
  "The Arabic must be natural Modern Standard Arabic and must preserve the EXACT meaning of the English " +
  "sentence you produced - translate meaning, not similar-sounding words. Before answering, double-check " +
  "that each Arabic word's root matches the intended meaning and is not a different, similar-sounding root " +
  "(e.g. Arabic مشى 'to walk' vs مشط 'to comb' are unrelated roots that must not be confused).";

const clean = (g) => (g.endsWith("1") ? g.slice(0, -1) : g);

const buildPrompt = (tokens) =>
  "Recognized signs, in order:\n" +
  tokens.map((cands, i) => `${i + 1}. ` + cands.map(([g, c]) => `${clean(g)} (${c.toFixed(2)})`).join(" | ")).join("\n");

function parse(text) {
  try {
    return JSON.parse(text.match(/\{[\s\S]*\}/)[0]);
  } catch {
    return { english: text.trim(), arabic: "" };
  }
}

async function gemini(prompt, env) {
  const model = env.GEMINI_MODEL || "gemini-2.5-flash";
  const r = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": env.GEMINI_API_KEY },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM }] },
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.2, responseMimeType: "application/json" },
      }),
    }
  );
  if (!r.ok) throw new Error(`gemini ${r.status}`);
  const j = await r.json();
  return j.candidates[0].content.parts.map((p) => p.text || "").join("");
}

async function chat(baseUrl, key, model, prompt) {
  const r = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: prompt },
      ],
    }),
  });
  if (!r.ok) throw new Error(`llm ${r.status}`);
  const j = await r.json();
  return j.choices[0].message.content;
}

const PROVIDERS = {
  gemini,
  groq: (p, env) =>
    chat("https://api.groq.com/openai/v1", env.GROQ_API_KEY, env.GROQ_MODEL || "llama-3.3-70b-versatile", p),
  openai: (p, env) => chat(env.LLM_BASE_URL, env.LLM_API_KEY, env.LLM_MODEL, p),
};

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "Content-Type": "application/json" } });

export async function onRequestPost({ request, env }) {
  let body;
  try {
    body = await request.json();
  } catch {
    return json({ detail: "bad json" }, 400);
  }
  const t = body && body.tokens;
  if (
    !Array.isArray(t) || t.length === 0 || t.length > 20 ||
    t.some((c) => !Array.isArray(c) || c.length === 0 || c.length > 5 ||
      c.some((x) => !Array.isArray(x) || typeof x[0] !== "string" || typeof x[1] !== "number"))
  ) {
    return json({ detail: "bad tokens" }, 400);
  }
  const tokens = t.map((cands) => cands.map(([g, c]) => [g.slice(0, 40), c]));

  const provider = (env.LLM_PROVIDER || "groq").toLowerCase();
  try {
    const text = await (PROVIDERS[provider] || PROVIDERS.groq)(buildPrompt(tokens), env);
    const out = parse(text);
    return json({ english: out.english || "", arabic: out.arabic || "", glosses: out.glosses });
  } catch (e) {
    return json({ english: tokens.map((c) => clean(c[0][0])).join(" ").toLowerCase(), arabic: "", error: String(e.message || e) });
  }
}

export const onRequestGet = () => json({ ok: true });
