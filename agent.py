"""Crop Doctor agent.

Two modes, same tools:
  * LLM mode  - if ANTHROPIC_API_KEY is set, an LLM decides which tools to call (true agent loop,
                supports photos and Kannada replies).
  * Offline   - no key needed; a rule-based agent runs the same tool pipeline and writes the report.
"""
import base64
import json
import mimetypes
import os

import tools
from rag import detect_crop

MODEL = os.environ.get("CROP_DOCTOR_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are Crop Doctor, an assistant for farmers in Coorg (Kodagu), Karnataka who grow coffee, pepper and cardamom.
Always work like this:
1. If a photo is given, first describe the visible symptoms in plain words.
2. Call disease_lookup with the symptoms (and the crop if known).
3. Call get_weather_risk for the most likely disease.
4. Call treatment_advice for that disease with a severity (low/medium/high) judged from the description.
5. Write the final answer with exactly these headings: Diagnosis, Why I think so, Weather risk, Treatment plan, Prevention checklist, When to call an expert.
Rules: use simple words a farmer understands. Use only the tool results for treatment advice; never invent pesticide doses.
If the match is weak, say you are not sure and ask for more details. Reply in the language requested."""

TOOL_SPECS = [
    {"name": "disease_lookup",
     "description": "Search the Coorg crop-disease knowledge base for diseases/pests matching the symptoms.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Symptom description in English"},
         "crop": {"type": "string", "enum": ["coffee", "pepper", "cardamom"]}},
         "required": ["query"]}},
    {"name": "get_weather_risk",
     "description": "Check current Coorg weather against the conditions that favour a disease; returns LOW/MEDIUM/HIGH risk.",
     "input_schema": {"type": "object", "properties": {"disease_id": {"type": "string"}},
                      "required": ["disease_id"]}},
    {"name": "treatment_advice",
     "description": "Get the treatment plan and prevention checklist for a disease.",
     "input_schema": {"type": "object", "properties": {
         "disease_id": {"type": "string"},
         "severity": {"type": "string", "enum": ["low", "medium", "high"]}},
         "required": ["disease_id"]}},
]

HIGH_WORDS = ["whole", "all plants", "many", "spreading", "spread fast", "severe", "dying", "dead", "everywhere", "entire"]
LOW_WORDS = ["few", "one plant", "just started", "small", "first", "slight", "early"]

EXPERT = ("Call the Kisan Call Centre (1800-180-1551) or visit your nearest Coffee Board extension office / "
          "Krishi Vigyan Kendra (KVK) if the problem spreads after treatment, if many plants are affected, or if you are unsure.")


def estimate_severity(text):
    t = (text or "").lower()
    if any(w in t for w in HIGH_WORDS):
        return "high"
    if any(w in t for w in LOW_WORDS):
        return "low"
    return "medium"


def confidence_label(score):
    if score >= 0.30:
        return "High"
    if score >= 0.18:
        return "Medium"
    return "Low"


class CropDoctorAgent:
    def __init__(self, weather_mode="auto"):
        self.weather_mode = weather_mode
        self.trace = []

    @property
    def llm_available(self):
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    # ------------------------------------------------------------ public entry
    def run(self, description, image_path=None, language="English"):
        self.trace = []
        description = (description or "").strip()
        note = ""
        if self.llm_available:
            try:
                return self._run_llm(description, image_path, language), self.trace
            except Exception as e:  # network / key problems -> fall back gracefully
                note = f"> LLM mode failed ({type(e).__name__}); showing the offline answer instead.\n\n"
                self.trace = []
        if image_path and not description:
            return ("Photo diagnosis needs LLM mode (set ANTHROPIC_API_KEY). "
                    "Please describe the problem in words, e.g. 'orange powder under coffee leaves'."), self.trace
        if language.lower().startswith("kan"):
            note += "> Kannada replies need LLM mode; showing English.\n\n"
        if image_path:
            note += "> Photo ignored in offline mode (needs LLM mode).\n\n"
        return note + self._run_offline(description), self.trace

    # ------------------------------------------------------------ tool dispatch
    def _call_tool(self, name, args):
        if name == "disease_lookup":
            out = tools.disease_lookup(args["query"], args.get("crop"))
        elif name == "get_weather_risk":
            out = tools.weather_risk(args["disease_id"], self.weather_mode)
        elif name == "treatment_advice":
            out = tools.treatment_advice(args["disease_id"], args.get("severity", "medium"))
        else:
            out = {"error": f"unknown tool {name}"}
        self.trace.append({"tool": name, "input": args, "output": out})
        return out

    # ------------------------------------------------------------ offline agent
    def _run_offline(self, text):
        if len(text.split()) < 2:
            return "Please describe the problem in a few words: which crop, which part (leaf, stem, berry) and what you see."
        crop = detect_crop(text)
        hits = self._call_tool("disease_lookup", {"query": text, **({"crop": crop} if crop else {})})
        top = hits[0]
        if top["score"] < 0.08:
            return ("I am not sure what this is from the description. Please tell me the crop and what you see "
                    "on the leaves, stems or berries (colour, spots, holes, smell). " + EXPERT)
        risk = self._call_tool("get_weather_risk", {"disease_id": top["disease_id"]})
        severity = estimate_severity(text)
        if risk["risk_level"] == "HIGH" and severity == "low":
            severity = "medium"  # favourable weather => treat as more urgent
        plan = self._call_tool("treatment_advice", {"disease_id": top["disease_id"], "severity": severity})

        lines = [f"## Diagnosis\n**{top['name']}** ({top['crop']}, {top['type']}), confidence: **{confidence_label(top['score'])}**\n"]
        lines.append("## Why I think so")
        if top["matched_symptoms"]:
            lines.append("You mentioned: " + ", ".join(top["matched_symptoms"]) + ".")
        lines.append(f"Known signs: {top['known_symptoms']}")
        lines.append(f"Cause: {top['cause']}.")
        others = [h["name"] for h in hits[1:] if h["score"] >= 0.08]
        if others:
            lines.append("Other possibilities: " + ", ".join(others) + ".")
        lines.append(f"\n## Weather risk\n**{risk['risk_level']}**: {risk['summary']}")
        for c in risk["checks"]:
            lines.append(f"- {'Yes' if c['met'] else 'No'}: {c['detail']}")
        lines.append(f"_Weather source: {risk['weather']['source']}_")
        lines.append(f"\n## Treatment plan\n_{plan['urgency']}_")
        lines += [f"{i}. {s}" for i, s in enumerate(plan["treatment_plan"], 1)]
        lines.append("\n## Prevention checklist")
        lines += [f"- [ ] {s}" for s in plan["prevention_checklist"]]
        lines.append(f"\n_{plan['safety_note']}_")
        lines.append(f"\n## When to call an expert\n{EXPERT}")
        return "\n".join(lines)

    # ------------------------------------------------------------ LLM agent loop
    def _run_llm(self, text, image_path, language):
        import anthropic

        client = anthropic.Anthropic()
        content = []
        if image_path:
            media = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            with open(image_path, "rb") as f:
                b64 = base64.standard_b64encode(f.read()).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}})
        content.append({"type": "text",
                        "text": f"Farmer's description: {text or '(photo only)'}\nReply language: {language}"})
        messages = [{"role": "user", "content": content}]

        for _ in range(8):
            resp = client.messages.create(model=MODEL, max_tokens=1800, system=SYSTEM_PROMPT,
                                          tools=TOOL_SPECS, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text")
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    out = self._call_tool(b.name, dict(b.input))
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(out, ensure_ascii=False)})
            messages.append({"role": "user", "content": results})
        return "The agent took too many steps. Please try again with a shorter description."


def format_trace(trace):
    if not trace:
        return "_No tools were called._"
    parts = []
    for i, t in enumerate(trace, 1):
        parts.append(f"**Step {i}: `{t['tool']}`**\n- input: `{json.dumps(t['input'], ensure_ascii=False)}`")
        out = t["output"]
        if t["tool"] == "disease_lookup":
            parts.append("- top matches: " + ", ".join(f"{h['name']} ({h['score']})" for h in out))
        elif t["tool"] == "get_weather_risk":
            parts.append(f"- {out.get('summary', out)}")
        else:
            parts.append(f"- plan for **{out.get('disease')}**, severity **{out.get('severity')}**")
    return "\n".join(parts)
