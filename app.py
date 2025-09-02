import os
import time

import pandas as pd
import shutil
import tempfile

import pdfplumber
import streamlit as st
import subprocess
from docx import Document
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from streamlit_pdf_viewer import pdf_viewer

from form_utils import extract_fields, fill_form, get_prompt_template, infer_data

st.set_page_config(page_title="Let It Fill", layout="wide", page_icon=":material/stylus_note:")
st.title("Let It Fill")

with st.expander("OpenAI API Key", expanded=True, icon=":material/vpn_key:"):
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "prev_api_key" not in st.session_state:
        st.session_state.prev_api_key = ""
    if "OPENAI_API_KEY" not in st.session_state:
        st.session_state.OPENAI_API_KEY = ""
    if "reset_api_key" not in st.session_state:
        st.session_state.reset_api_key = False

    if "show_form" not in st.session_state:
        st.session_state.show_form = False

    if "show_inference" not in st.session_state:
        st.session_state.show_inference = False

    if "show_download" not in st.session_state:
        st.session_state.show_download = False

    if st.session_state.reset_api_key:
        st.session_state.openai_api_key = ""
        st.session_state.reset_api_key = False

    openai_api_key = st.text_input("Introduce tu API Key", type="password", key="openai_api_key")

    _, col1, col2 = st.columns([16.00, 0.86, 1.00])  # "wide": [16.00, 0.86, 1.00]; "centered": [5.30, 0.85, 0.97]
    with col1:
        delete_api_key_button = st.button(label="Borrar", key="delete_api_key_button", help="Pulsa para borrar la OpenAI API Key.", type="secondary")
    with col2:
        save_api_key_button = st.button(label="Guardar", key="save_api_key_button", help="Pulsa para guardar la OpenAI API Key.", type="primary")

    if save_api_key_button or (openai_api_key and openai_api_key != st.session_state.prev_api_key):
        if not openai_api_key.strip():
            st.warning("Por favor, introduce tu API Key antes de continuar.", icon=":material/warning:")
        elif not openai_api_key.startswith("sk-"):
            st.error("La clave introducida debe comenzar por 'sk-'.", icon=":material/error:")
        else:
            st.session_state["OPENAI_API_KEY"] = openai_api_key
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.session_state.prev_api_key = openai_api_key

            placeholder = st.empty()
            placeholder.success("API Key guardada correctamente.", icon=":material/check_circle:")
            time.sleep(2)
            placeholder.empty()

            st.session_state.show_form = True

    if delete_api_key_button:
        st.session_state["OPENAI_API_KEY"] = ""
        st.session_state["prev_api_key"] = ""
        st.session_state["reset_api_key"] = True
        os.environ["OPENAI_API_KEY"] = ""
        st.rerun()

if st.session_state.show_form:
    with st.expander("Entrada de información", expanded=True, icon=":material/docs:"):
        with st.form("user_form", border=False):
            tab1, tab2, tab3 = st.tabs(["Documento", "Datos personales", "Modelo"])

            with tab1:
                user_file = st.file_uploader("Sube un documento a rellenar", type=["pdf", "docx"])

            with tab2:
                user_input = st.text_area("Describe tu información personal", height=250, placeholder="Ej: Mi nombre es...")

            with tab3:
                model_options = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4.1"]
                st.session_state["openai_model"] = st.radio(
                    "Selecciona el modelo a usar para la inferencia",
                    options=model_options,
                    index=model_options.index("gpt-4o-mini")
                )

            _, col3 = st.columns([18.35, 1.00])  # "wide": [18.35, 1.00]; "centered": [6.89, 1.00]
            with col3:
                user_form_submit_button = st.form_submit_button(label="Aceptar", help="Pulsa para enviar el formulario y tus datos.", type="primary")

        openai_api_key_changed = st.session_state.get("OPENAI_API_KEY") != openai_api_key
        user_file_changed = st.session_state.get("user_file") != user_file
        user_input_changed = st.session_state.get("user_input") != user_input

        if user_form_submit_button:
            if not user_input.strip():
                st.warning("Introduce tu información personal.", icon=":material/warning:")
            elif not user_file:
                st.warning("Sube un archivo válido.", icon=":material/warning:")
            else:
                st.session_state["user_file"] = user_file
                st.session_state["user_input"] = user_input

                if openai_api_key_changed or user_file_changed or user_input_changed:
                    st.session_state.inference_done = False

                st.session_state.show_inference = True

if st.session_state.show_inference:
    with st.expander("Inferencia", expanded=True, icon=":material/model_training:"):

        if "inference_done" not in st.session_state:
            st.session_state["inference_done"] = False

        if "llm" not in st.session_state or st.session_state.get("llm_api_key") != st.session_state.OPENAI_API_KEY:
            st.session_state["llm"] = ChatOpenAI(api_key=st.session_state["OPENAI_API_KEY"], model=st.session_state.openai_model, temperature=0, seed=4567)
            st.session_state["llm_api_key"] = st.session_state.OPENAI_API_KEY

        if not st.session_state.inference_done:
            extension = os.path.splitext(st.session_state.user_file.name)[-1].lower()
            base_filename = os.path.splitext(st.session_state.user_file.name)[0]

            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_input_file:
                tmp_input_file.write(st.session_state.user_file.read())
                input_path = tmp_input_file.name

            st.session_state["input_path"] = input_path
            st.session_state["extension"] = extension
            st.session_state["base_filename"] = base_filename

            system_template, user_template = get_prompt_template(st.session_state.extension)
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(user_template)
            ])
            
            chain = prompt | st.session_state.llm

            with st.spinner("Extrayendo campos del documento...", show_time=True):
                fields = extract_fields(st.session_state.input_path, st.session_state.extension)

            if not fields:
                if st.session_state.extension == ".pdf":
                    doc = Document()
                    with pdfplumber.open(st.session_state.input_path) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                doc.add_paragraph(text)
                                doc.add_page_break()
                    pdf2docx_path = st.session_state.input_path.replace(".pdf", ".docx")
                    doc.save(pdf2docx_path)

                    st.session_state.input_path = pdf2docx_path
                    st.session_state.extension = ".docx"

                    with st.spinner("Extrayendo campos del documento...", show_time=True):
                        fields = extract_fields(st.session_state.input_path, st.session_state.extension)

            if not fields:
                st.error("No existen campos a rellenar en el documento.")

            else:
                with st.spinner("Infiriendo valores...", show_time=True):
                    inferred_data = infer_data(fields, user_input, chain)

                st.session_state["fields"] = fields
                st.session_state["inferred_data"] = inferred_data
                st.session_state["editable_df"] = pd.DataFrame(inferred_data.items(), columns=["Campo", "Valor"])
                st.session_state["edited_data"] = inferred_data.copy()

            st.session_state.inference_done = True

        with st.popover(label="ChatBot", icon=":material/chat:", use_container_width=True):
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            user_prompt = st.chat_input("Pregúntame lo que quieras")

            if user_prompt:
                st.session_state.chat_messages.append({"role": "user", "content": user_prompt})

                user_context = st.session_state.get("user_input", "")

                inferred_context = ""
                if "inferred_data" in st.session_state:
                    inferred_context = "\n".join([f"{k}: {v}" for k, v in st.session_state.inferred_data.items()])

                assistant_prompt = f"""
You are provided with the following context regarding the document and the user:

User description:
{user_context}

Inferred fields and values:
{inferred_context}

The user asks:
{user_prompt}

Respond in a clearly, concisely, thoroughly, and accurately.
""".strip()
        
                response = st.session_state.llm.invoke([HumanMessage(content=assistant_prompt)])
                assistant_text = response.content

                st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})

            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        if "inferred_data" in st.session_state and "editable_df" not in st.session_state:
            st.session_state.editable_df = pd.DataFrame(st.session_state.inferred_data.items(), columns=["Campo", "Valor"])

        if "editable_df" in st.session_state:
            edited_df = st.data_editor(
                st.session_state.editable_df,
                num_rows="fixed",
                use_container_width=True,
                key="edited_df"
            )

            if not edited_df.equals(st.session_state.editable_df):
                edited_data = edited_df.set_index("Campo")["Valor"].to_dict()
                st.session_state["edited_data"] = edited_data

            default_filename = f"{st.session_state.base_filename}_completed_{st.session_state.openai_model}{st.session_state.extension}"

            if "final_filename" not in st.session_state:
                st.session_state["final_filename"] = default_filename

            st.session_state.final_filename = st.text_input("Documento rellenado", value=st.session_state.final_filename, icon=":material/edit_square:")

            if not st.session_state.final_filename.strip():
                st.session_state["final_filename"] = default_filename

            _, col4 = st.columns([14.71, 1.00])  # "wide": [14.71, 1.00]; "centered": [5.35, 1.00]
            with col4:
                complete_document_button = st.button(label="Completar", key="complete_document_button", help="Pulsa para completar el documento.", type="primary")

            if complete_document_button:
                st.session_state.show_download = True

if st.session_state.show_download:
    with st.expander("Descarga", expanded=True, icon=":material/file_save:"):
        preview_bytes = None
        download_bytes = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=st.session_state.extension) as tmp_output_file:
            output_path = tmp_output_file.name

        with st.spinner("Completando documento...", show_time=True):
            fill_form(
                st.session_state.input_path,
                output_path,
                st.session_state.edited_data,
                st.session_state.fields,
                st.session_state.extension
            )

        with open(output_path, "rb") as f:
            download_bytes = f.read()

        if st.session_state.extension == ".docx":
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            mime_type = "application/pdf"

        preview_bytes = None

        if st.session_state.extension == ".docx":
            with tempfile.TemporaryDirectory() as tmp_dir:
                docx_copy_path = os.path.join(tmp_dir, os.path.basename(output_path))
                shutil.copy(output_path, docx_copy_path)

                try:
                    subprocess.run([
                        "libreoffice",
                        "--headless",
                        "--convert-to", "pdf",
                        "--outdir", tmp_dir,
                        docx_copy_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    pdf_path = os.path.join(tmp_dir, os.path.splitext(os.path.basename(output_path))[0] + ".pdf")
                    with open(pdf_path, "rb") as f:
                        preview_bytes = f.read()

                except Exception as e:
                    st.error(f"An error occurred while converting DOCX to PDF with LibreOffice: {e}")
        elif st.session_state.extension == ".pdf":
            preview_bytes = download_bytes

        if preview_bytes:
            eye_svg = """
                <svg xmlns="http://www.w3.org/2000/svg" height="24" width="24" style="vertical-align: middle">
                <path d="M12 6.5c-3.55 0-6.39 2.23-7.5 5.5 1.11 3.27 3.95 5.5 7.5 5.5s6.39-2.23 7.5-5.5c-1.11-3.27-3.95-5.5-7.5-5.5zm0 9a3.5 3.5 0 1 1 0-7 3.5 3.5 0 0 1 0 7z"/>
                <circle cx="12" cy="12" r="2"/>
                </svg>
            """
            st.markdown(f"<h3>{eye_svg} Vista previa</h3>", unsafe_allow_html=True)

            pdf_viewer(
                input=preview_bytes,
                width="100%",
                height=1000,
                zoom_level=1.0,
                render_text=True,
                show_page_separator=False
            )

        _, col5 = st.columns([15.34, 1.00])  # "wide": [15.34, 1.00]; "centered": [5.35, 1.00]
        with col5:
            st.download_button(
                label="Descargar",
                data=download_bytes,
                file_name=st.session_state.final_filename,
                mime=mime_type,
                help="Pulsa para descargar el documento completado.",
                type="primary"
            )