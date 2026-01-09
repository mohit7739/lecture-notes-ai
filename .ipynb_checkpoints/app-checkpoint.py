import streamlit as st
import whisper
import os
import google.generativeai as genai
import tempfile

# --- CONFIGURATION ---
st.set_page_config(page_title="Lecture Voice-to-Notes", page_icon="🎓")

# --- API KEY HANDLING (The Smart Way) ---
# 1. Try to get key from "secrets.toml" (Local) or "Streamlit Cloud Secrets"
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    # 2. If no secret is found, show the sidebar input (Backup plan)
    with st.sidebar:
        api_key = st.text_input("Enter Google Gemini API Key", type="password")

# Warning if no key is found at all
if not api_key:
    st.error("No API Key found! Please add it to .streamlit/secrets.toml or the sidebar.")
    st.stop() # Stop the app here

# --- APP TITLE ---
st.title("🎓 Lecture Voice-to-Notes AI")
# ... (Rest of your code remains the same)

# --- MAIN LOGIC ---
uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a"])

if uploaded_file is not None and api_key:
    st.audio(uploaded_file, format="audio/mp3")
    
    if st.button("Generate Notes"):
        with st.spinner("Processing... (This takes a minute)"):
            try:
                # 1. Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_filename = tmp_file.name

                # 2. Transcribe with Whisper
                st.info("Step 1/2: Listening to the lecture... 👂")
                
                # Load model (cached so it doesn't reload every time)
                @st.cache_resource
                def load_whisper():
                    return whisper.load_model("base")
                
                model = load_whisper()
                result = model.transcribe(tmp_filename)
                transcription = result["text"]

                # 3. Summarize with Gemini
                st.info("Step 2/2: Writing notes... ✍️")
                genai.configure(api_key=api_key)
                
                # Use the stable model alias
                model_gemini = genai.GenerativeModel('models/gemini-flash-latest')
                
                prompt = f"""
                You are an expert student assistant. 
                Take this lecture transcript and convert it into:
                1. A Bulleted Summary
                2. 5 Key Study Questions (with answers at the bottom)
                
                Transcript: {transcription}
                """
                
                # RETRY LOGIC: Tries 3 times if it fails
                import time
                for attempt in range(3):
                    try:
                        response = model_gemini.generate_content(prompt)
                        break # If successful, stop looping
                    except Exception as e:
                        if "429" in str(e):
                            st.warning(f"Traffic is high. Retrying in 5 seconds... (Attempt {attempt+1}/3)")
                            time.sleep(5)
                        else:
                            raise e # If it's a real error, crash
                prompt = f"""
                You are an expert student assistant. 
                Take this lecture transcript and convert it into:
                1. A Bulleted Summary
                2. 5 Key Study Questions (with answers at the bottom)
                
                Transcript: {transcription}
                """
                
                response = model_gemini.generate_content(prompt)
                
                # 4. Display Result
                st.success("Done!")
                st.markdown("### 📝 Study Notes")
                st.markdown(response.text)

                # Cleanup
                os.remove(tmp_filename)

            except Exception as e:
                st.error(f"An error occurred: {e}")

elif uploaded_file and not api_key:
    st.warning("Please enter your Google Gemini API Key in the sidebar to proceed.")