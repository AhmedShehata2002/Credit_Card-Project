import streamlit as st
import pdfplumber
import openai
import os

# Set Streamlit page configuration
st.set_page_config(page_title="PDF Uploader & Data Chat")

# File upload section
uploaded_file = st.file_uploader(label="📂 Upload your PDF file", type="pdf")

if uploaded_file is not None:
    try:
        # Extract text from the uploaded PDF
        with pdfplumber.open(uploaded_file) as pdf:
            extracted_text = ""
            for page in pdf.pages:
                extracted_text += page.extract_text() + "\n"

        if not extracted_text.strip():
            st.error("No text could be extracted from the uploaded PDF.")
            st.stop()

        st.write("### Extracted Text Preview:")
        st.text_area("Extracted Text", value=extracted_text, height=300)

        # Interaction with the extracted text
        st.write("# 🧘 Talk to Your Data:")
        query = st.text_area("🗣️ Ask a question about the text in the PDF")

        if query:
            # Retrieve OpenAI API key from environment variable
            try:
                openai.api_key = os.getenv("OPENAI_API_KEY")
                if not openai.api_key:
                    st.error("Please set the OPENAI_API_KEY environment variable in your terminal.")
                    st.stop()

                # Query OpenAI API
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an assistant skilled at analyzing text data."},
                        {"role": "user", "content": f"Here is the extracted text:\n{extracted_text}\n\n{query}"}
                    ],
                )

                # Display the ChatGPT response
                st.write("### ChatGPT Response:")
                st.write(response["choices"][0]["message"]["content"])

            except Exception as e:
                # Handle any errors from the OpenAI API
                st.error(f"Error interacting with ChatGPT API: {e}")

    except Exception as e:
        st.error(f"Error processing the PDF: {e}")
