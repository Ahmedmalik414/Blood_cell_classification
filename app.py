import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("blood_cell_best.h5")
class_names = ['Basophil', 'Erythroblast', 'Monocyte', 'Myeloblast', 'Segmented Neutrophil']

def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img):
    img_processed = preprocess_image(img)
    predictions = model.predict(img_processed)[0]
    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

custom_css = """
body {
    background: linear-gradient(to right, #e0f7fa, #fce4ec);
}
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
    color: #1a237e;
}
h1, h3 {
    text-align: center;
    color: #01579b;
}
#upload_area {
    border: 3px dashed #4fc3f7;
    border-radius: 12px;
    background-color: #ffffffee;
    padding: 20px;
}
.gr-button {
    background-color: #26c6da;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    transition: 0.3s ease;
}
.gr-button:hover {
    background-color: #00acc1;
}
.label {
    background-color: #ffffffdd;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 0 8px rgba(0,0,0,0.1);
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("""
    <div style='text-align: center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/3774/3774299.png' width='80'/>
    </div>
    <h1>🩸 Blood Cell Classifier</h1>
    <h3>Classify blood cells using a deep learning model trained on microscopic images</h3>
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="🔬 Upload Blood Cell Image", elem_id="upload_area")
            submit_btn = gr.Button("🔍 Analyze Cell", elem_classes="gr-button")
        with gr.Column():
            label_output = gr.Label(label="📊 Prediction Results", elem_classes="label")

    submit_btn.click(fn=predict, inputs=image_input, outputs=label_output)

    gr.Markdown("""
    ---
    <div style='text-align: center;'>
        <img src='https://cdn-icons-png.flaticon.com/512/2883/2883841.png' width='60'/>
    </div>
    <h3>🧬 About This App</h3>
    <p style="text-align:center; max-width: 700px; margin: auto;">
        This AI-powered app uses a ResNet50 model to classify blood cells into Basophil, Erythroblast, Monocyte, Myeloblast, or Segmented Neutrophil.
        It's built to assist medical professionals, students, and researchers in diagnostics and education.
    </p>
    """)

demo.launch()
