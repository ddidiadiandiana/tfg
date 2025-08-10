import json
from datetime import date
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st
from docx import Document
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject

from langchain.chains import LLMChain

def build_mapping(extracted_fields: List[Dict[str, str]], inferred_data: Dict[str, str]) -> Dict[str, Optional[str]]:
    """
    Map extracted field names to inferred field names based on stripped equality.

    :param extracted_fields: List of dicts with 'name' keys for extracted fields.
    :param inferred_data: Dict of inferred field names to values.
    :return: Dictionary mapping extracted field names to inferred field names, or None if no match.
    """
    mapping: Dict[str, Optional[str]] = {}

    for field in extracted_fields:
        extracted_name = field.get("name", "").strip()
        match = None
        for inferred_name in inferred_data.keys():
            if extracted_name == inferred_name.strip():
                match = inferred_name
                break
        mapping[field.get("name", "")] = match

    return mapping

def get_prompt_template(extension: str) -> Tuple[str, str]:
    """
    Returns system and user prompt templates based on the file extension.

    :param extension: File extension indicating document type.
    :return: Tuple containing (system_template, user_template) strings for prompting the language model.
    """    
    if extension == ".docx":
        system_template = """
You are an expert assistant in filling out forms accurately and professionally.
""".strip()

        user_template = """
Today's date is {today}.

Instructions:
- Use the user input to complete all required fields.
- Match expected formats carefully (name order eg "Surname, Name"; case sensitivity and accents eg "gran via" → "Gran Vía"; date formats eg "1 de enero de 2025" → "01/01/2025").
- Use uppercase letters in IDs, bank accounts, and similar fields.
- Do not repeat currency symbols or labels if they already appear in the document.
- Use {today} date if the user mentions "today"/"date" (case-insensitive and in any language). Infer the end date if the user provides a start date and a duration.
- Ask for any missing info using natural language.

Given the following fields:
{fields}

And the user input:
<<< {user_input} >>>

Return only a JSON dictionary with keys that exactly match the field names.
""".strip()

    elif extension == ".pdf":
        system_template = """
You are an expert assistant in filling out forms accurately and professionally.
""".strip()
        
        user_template = """
Today's date is {today}.

Instructions:
- Use the user input to complete all required fields.
- Each field has a name, dimensions, and position.
- Match expected formats carefully (name order eg "Surname, Name"; case sensitivity and accents eg "gran via" → "Gran Vía"; date formats eg "1 de enero de 2025" → "01/01/2025").
- Use uppercase letters in IDs, bank accounts, and similar fields.
- Do not repeat currency symbols or labels if they already appear in the document.
- Use {today} date if the user mentions "today"/"date" (case-insensitive and in any language). Infer the end date if the user provides a start date and a duration.
- Ask for any missing info using natural language.

Fields: {fields}

User input: <<< {user_input} >>>

Return only a JSON dictionary with keys that exactly match the field names.
""".strip()

    return system_template, user_template

def extract_fields(file: str, extension: str) -> List[Dict[str, Any]]:
    """
    Extracts candidate form fields from a DOCX or PDF file.

    :param file: Path to the input file.
    :param extension: File extension indicating document type.
    :return: List of dictionaries representing detected fields and their metadata.
    """
    if extension == ".docx":
        doc = Document(file)
        fields = []
    
        # Paragraph fields
        for paragraph_i, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            if text and (":" in text or "-" in text or len(text.split()) <= 6):
                fields.append({
                    "name": text,
                    "location": "paragraph",
                    "paragraph": paragraph_i,
                    "append_style": "inline"
                })
    
        # Table fields
        for table_i, table in enumerate(doc.tables):
            for row_i, row in enumerate(table.rows):
                for column_i, cell in enumerate(row.cells):
                    field_name = cell.text.strip()
                    if field_name and len(field_name.split()) <= 6:
                        append_style = "inline"
                        if column_i + 1 < len(row.cells) and not row.cells[column_i + 1].text.strip():
                            append_style = "cell_right"
                        elif row_i + 1 < len(table.rows) and not table.rows[row_i + 1].cells[column_i].text.strip():
                            append_style = "cell_below"
    
                        fields.append({
                            "name": field_name,
                            "location": "table",
                            "table": table_i,
                            "row": row_i,
                            "column": column_i,
                            "append_style": append_style
                        })
        
    elif extension == ".pdf":
        reader = PdfReader(file)
        fields = []
    
        for page_num, page in enumerate(reader.pages, start=1):
            annots = page.get("/Annots")
            if annots is not None:
                for annotation in list(annots):
                    obj = annotation.get_object()
                    name = str(obj.get("/T")) if isinstance(obj.get("/T"), TextStringObject) else ""
                    value = str(obj.get("/V")) if isinstance(obj.get("/V"), TextStringObject) else ""
                    rect = obj.get("/Rect", [0, 0, 0, 0])
    
                    try:
                        width = float(rect[2]) - float(rect[0])
                        height = float(rect[3]) - float(rect[1])
                    except Exception:
                        width, height = 0, 0
    
                    fields.append({
                        "name": name,
                        "default_value": value,
                        "width": width,
                        "height": height,
                        "position": rect,
                        "page": page_num
                    })

    return fields

def extract_json_dict(text: str) -> str:
    """
    Extracts the first JSON-like dictionary from a string.

    :param text: The input string potentially containing a JSON object.
    :return: A string containing a JSON block, or an empty string if none is found.
    """

    stack = []
    start = None

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start = i
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    return text[start:i+1]

    return ""

def infer_data(fields: List[Dict[str, Any]], user_input: str, chain: LLMChain) -> Dict[str, str]:
    """
    Infers field values from user input using a language model chain.
    
    :param fields: List of dictionaries containing field names and their properties.
    :param user_input: Text input providing relevant personal information from the user.
    :param chain: LangChain LLMChain used to invoke the language model.
    :param extension: File extension indicating document type.
    :return: Dictionary mapping field names to inferred values.
    """
    fields = [field["name"] for field in fields]
    
    prompt_vars = {
        "today": date.today().strftime("%d/%m/%Y"),
        "fields": json.dumps(fields, indent=2, ensure_ascii=False),
        "user_input": user_input
    }

    response = chain.invoke(prompt_vars).content.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        extracted_json = extract_json_dict(response)
        try:
            return json.loads(extracted_json)
        except json.JSONDecodeError:
            st.error("No se ha podido interpretar la respuesta del modelo como un JSON válido. El proceso se ha detenido.", icon=":material/error:")
            st.stop()

def fill_form(inputf: str, outputf: str, data: Dict[str, str], fields: List[Dict[str, Any]], extension: str, mapping: Dict[str, str] = None) -> None:
    """
    Fills document form fields with provided data.
    
    :param inputf: Path to the input file to be filled.
    :param outputf: Path to save the filled output document.
    :param data: Dictionary associating field names with fill values.
    :param fields: List of field metadata extracted from the document.
    :param extension: File extension indicating document type.
    :param mapping: Dictionary mapping original field names to inferred field names.
    """
    if extension == ".docx":
        doc = Document(inputf)

        for field in fields:
            extracted_key = field["name"]
            inferred_key = mapping.get(extracted_key) if mapping else extracted_key

            if not inferred_key:
                continue

            value = data.get(inferred_key, "")
            if not value:
                continue

            # Fill paragraphs
            if field["location"] == "paragraph":
                paragraph = doc.paragraphs[field["paragraph"]]
                if extracted_key in paragraph.text:
                    paragraph.text = paragraph.text.replace(extracted_key, f"{extracted_key} {value}")
    
            # Fill tables
            elif field["location"] == "table":
                table = doc.tables[field["table"]]
                row, column = field["row"], field["column"]

                if field["append_style"] == "cell_right":
                    table.cell(row, column + 1).text = value
                elif field["append_style"] == "cell_below":
                    table.cell(row + 1, column).text = value
                else:
                    table.cell(row, column).text = f"{extracted_key} {value}"
    
        doc.save(outputf)
        
    elif extension == ".pdf":
        reader = PdfReader(inputf)

        for page in reader.pages:
            annots = page.get("/Annots")
            if annots is not None:
                for annotation in list(annots):
                    obj = annotation.get_object()
                    name = obj.get("/T")
    
                    if isinstance(name, TextStringObject):
                        field_name = str(name)
                    else:
                        field_name = ""
    
                    value = data.get(field_name, "")
    
                    if value:
                        obj.update({NameObject("/V"): TextStringObject(str(value))})
    
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
    
        writer.write(outputf)