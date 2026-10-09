"""Crop Doctor for Coorg Farmers - Gradio app.  Run: python app.py"""
import gradio as gr

from agent import CropDoctorAgent, format_trace

EXAMPLES = [
    ["Orange powder under my coffee leaves and many leaves are falling", None, "English", "Auto (live weather)"],
    ["Pepper vine suddenly wilted, black spots on leaves and rotting at the base", None, "English", "Auto (live weather)"],
    ["Cardamom leaves have pale streaks along the veins and the clump is stunted", None, "English", "Dry season"],
    ["Coffee leaves turned black and are hanging on threads after continuous rain", None, "English", "Monsoon-like"],
]
WEATHER_MODES = {"Auto (live weather)": "auto", "Monsoon-like": "monsoon", "Dry season": "dry"}


def diagnose(description, image_path, language, weather_choice):
    agent = CropDoctorAgent(weather_mode=WEATHER_MODES.get(weather_choice, "auto"))
    report, trace = agent.run(description, image_path, language)
    return report, format_trace(trace)


def build_ui():
    probe = CropDoctorAgent()
    mode = "LLM agent mode (photo + Kannada enabled)" if probe.llm_available else "Offline mode (no API key needed)"
    with gr.Blocks(title="Crop Doctor for Coorg Farmers") as demo:
        gr.Markdown("# Crop Doctor for Coorg Farmers\n"
                    "Describe what you see on your **coffee, pepper or cardamom** plants and get a diagnosis, "
                    f"treatment plan and prevention checklist.\n\n_Running in: **{mode}**_")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Describe the problem", lines=4,
                                  placeholder="e.g. orange powder under coffee leaves, leaves falling")
                image = gr.Image(label="Leaf photo (optional, needs LLM mode)", type="filepath")
                language = gr.Dropdown(["English", "Kannada"], value="English", label="Reply language")
                weather = gr.Dropdown(list(WEATHER_MODES), value="Auto (live weather)", label="Weather source")
                btn = gr.Button("Diagnose", variant="primary")
            with gr.Column():
                report = gr.Markdown(label="Report")
        with gr.Accordion("Agent steps (which tools were called)", open=False):
            trace = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=[text, image, language, weather])
        btn.click(diagnose, [text, image, language, weather], [report, trace])
        gr.Markdown("_Advice is for guidance only. Confirm chemical use with your Coffee Board / Spices Board officer or KVK._")
    return demo


if __name__ == "__main__":
    build_ui().launch(share=True)
