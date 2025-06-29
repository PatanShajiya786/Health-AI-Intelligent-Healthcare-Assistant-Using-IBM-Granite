from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr
from PIL import Image
import numpy as np
import io

# Load the model
model_id = "ibm-granite/granite-3.3-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
model.to("cpu")  # Safer for deployment

# Query the model
def query_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Backend logic
def health_chat(message):
    try:
        response = query_model(f"You are a compassionate health assistant. {message}")
        return str(response).strip() if response else "I'm not sure how to respond—can you rephrase?"
    except Exception as e:
        return f"⚠️ Sorry, something went wrong: {str(e)}"

def predict_disease(symptoms):
    return query_model(f"What disease could be associated with these symptoms: {symptoms}?")

def home_remedy(condition):
    return query_model(f"What are natural home remedies for {condition}?")

def generate_treatment_plan(condition):
    return query_model(f"Provide a treatment plan for {condition}.")

def visualize_and_analyze():
    img = Image.fromarray(np.uint8(np.random.rand(200, 300, 3) * 255))
    return img, query_model("Analyze this sample patient data.")

def update_profile(name, age, gender, notes):
    return f"✅ Profile Saved:\nName: {name}\nAge: {age}\nGender: {gender}\nNotes: {notes}"

# Custom CSS
custom_css = """
#sidebar-container {
    background: linear-gradient(to bottom, #34226d, #3c3b92, #4e6cb5);
    padding: 20px;
    border-radius: 10px;
    color: white;
}
#main-area {
    background: linear-gradient(135deg, #ff69b4, #8a2be2, #1e90ff);
    padding: 20px;
    border-radius: 12px;
    color: white;
}
"""

# Gradio Interface
with gr.Blocks(title="HealthAI", css=custom_css, theme=gr.themes.Soft()) as healthai:
    gr.Markdown("## 🧠 HealthAI: Your AI-Powered Health Companion")

    with gr.Row():
        with gr.Column(scale=1, elem_id="sidebar-container"):
            selector = gr.Radio(
                choices=[
                    "Patient Chat", "Disease Prediction", "Home Remedies",
                    "Treatment Generator", "Health Analytics", "Profile Manager"
                ],
                label="📋 Choose a Module",
                value="Patient Chat",
                interactive=True
            )

        with gr.Column(scale=3, elem_id="main-area"):
            with gr.Column(visible=True) as chat_ui:
                chat_input = gr.Textbox(label="You")
                chat_output = gr.Textbox(label="Health Assistant's Reply")
                chat_input.submit(fn=health_chat, inputs=chat_input, outputs=chat_output)

            with gr.Column(visible=False) as prediction_ui:
                gr.Interface(fn=predict_disease,
                             inputs=gr.Textbox(label="Symptoms"),
                             outputs=gr.Textbox(label="Predicted Diseases", lines=12))

            with gr.Column(visible=False) as remedy_ui:
                gr.Interface(fn=home_remedy,
                             inputs=gr.Textbox(label="Condition or Symptom"),
                             outputs=gr.Textbox(label="Home Remedy", lines=10))

            with gr.Column(visible=False) as treatment_ui:
                gr.Interface(fn=generate_treatment_plan,
                             inputs=gr.Textbox(label="Condition"),
                             outputs=gr.Textbox(label="Treatment Plan", lines=10))

            with gr.Column(visible=False) as analytics_ui:
                gr.Interface(fn=visualize_and_analyze,
                             inputs=[],
                             outputs=[gr.Image(), gr.Textbox(label="AI Insights", lines=8)])

            with gr.Column(visible=False) as profile_ui:
                gr.Interface(fn=update_profile,
                             inputs=[
                                 gr.Textbox(label="Name"),
                                 gr.Number(label="Age"),
                                 gr.Radio(["male", "female", "other"], label="Gender"),
                                 gr.Textbox(label="Medical Notes")
                             ],
                             outputs=gr.Textbox(label="Saved Profile", lines=8))

    def toggle(selected):
        return {
            chat_ui: gr.update(visible=selected == "Patient Chat"),
            prediction_ui: gr.update(visible=selected == "Disease Prediction"),
            remedy_ui: gr.update(visible=selected == "Home Remedies"),
            treatment_ui: gr.update(visible=selected == "Treatment Generator"),
            analytics_ui: gr.update(visible=selected == "Health Analytics"),
            profile_ui: gr.update(visible=selected == "Profile Manager"),
        }

    selector.change(fn=toggle, inputs=selector, outputs=[
        chat_ui, prediction_ui, remedy_ui, treatment_ui, analytics_ui, profile_ui
    ])

healthai.launch()