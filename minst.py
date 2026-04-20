import streamlit as st
import numpy as np
from PIL import Image
import joblib
from streamlit_drawable_canvas import st_canvas
import scipy.ndimage as ndi

# Load model + scaler
model = joblib.load("mnist_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Digit Recognizer", layout="centered")

st.title("🧠 Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) and click Predict")

# Canvas (thin strokes = better)
canvas = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Centering function
def center_image(img):
    cy, cx = ndi.center_of_mass(img)
    if np.isnan(cx) or np.isnan(cy):
        return img
    shiftx = int(np.round(img.shape[1]/2 - cx))
    shifty = int(np.round(img.shape[0]/2 - cy))
    return ndi.shift(img, [shifty, shiftx])

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear"):
        st.rerun()

with col2:
    if st.button("🔮 Predict"):
        if canvas.image_data is not None:

            # Get image
            img = canvas.image_data

            # Convert to grayscale (0–255 like MNIST)
            img = Image.fromarray((img[:, :, 0]).astype(np.uint8))

            # Resize to 28x28
            img = img.resize((28, 28))

            # Convert to numpy (IMPORTANT: float64 like training)
            img_array = np.array(img).astype(np.float64)


            # Flatten
            img_flat = img_array.reshape(1, -1)

            # Apply SAME scaler used in training
            img_scaled = scaler.transform(img_flat)

            # Predict
            prediction = model.predict(img_scaled)

            # Display image safely
            st.image(img_array, caption="Processed Image", width=150, clamp=True)

            st.success(f"🎯 Prediction: {prediction[0]}")