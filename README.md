# Crop Doctor for Coorg Farmers

An AI agent that helps Coorg farmers identify **coffee, pepper and cardamom** diseases/pests and tells them what to do.

**Input:** text description (and optionally a leaf photo)
**Agent brain:** RAG over a knowledge base of 8 Coorg crop problems (TF-IDF retrieval) + optional LLM
**Tools:** `disease_lookup`, `get_weather_risk`, `treatment_advice`
**Output:** diagnosis, weather risk, treatment plan, prevention checklist

## Run on Google Colab (easiest)
1. Upload `crop_doctor.zip` to Colab and open `CropDoctor_Colab.ipynb` (inside the zip), or just run the cells in it.
2. Run all cells; click the public link shown by Gradio.

## Run locally
```
pip install -r requirements.txt
python test_run.py     # 8 sample cases
python app.py          # web UI
```

## Two modes (same tools)
| Mode | How to enable | Features |
|------|---------------|----------|
| Offline | nothing needed | text input, English, rule-based agent calls the 3 tools |
| LLM agent | set `ANTHROPIC_API_KEY` | LLM chooses the tool calls, photo input, Kannada replies |

## Project structure
```
app.py            Gradio UI
agent.py          agent loop (offline + LLM)
tools.py          disease_lookup, weather_risk, treatment_advice, live weather (Open-Meteo)
rag.py            TF-IDF knowledge-base retrieval
data/diseases.json  knowledge base (add your own diseases here)
test_run.py       self-test
```

## Architecture
Farmer text/photo -> Agent -> disease_lookup (RAG) -> get_weather_risk (live Coorg weather vs. disease conditions) -> treatment_advice -> report

## 2-minute demo script
1. Problem: farmers lose crops because they identify diseases late.
2. Type: "Orange powder under my coffee leaves and many leaves are falling".
3. Show the report, then open "Agent steps" to show the three tool calls.
4. Switch weather to "Dry season" and run the pepper example to show the risk level change.
5. Future scope: photo classifier trained on leaf images, WhatsApp bot, voice input in Kannada, more diseases.

## Notes and limits
- Knowledge base is a demo set; verify advice with the Coffee Board / Spices Board / KVK before real use.
- No pesticide doses are given on purpose; the app tells farmers to follow label and expert advice.
- Live weather uses the free Open-Meteo API; if it is unreachable the app estimates from the season.
