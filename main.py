from google.adk.agents import Agent, InvocationContext

from google.adk.sessions.session import Session
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import asyncio
import os
from pathlib import Path
import sys
import pdfplumber
import uuid

load_dotenv()

# Use the updated model specification pattern
model = "gemini-2.5-pro"  # Updated to native Gemini model


# Pydantic Models
class DocumentSummary(BaseModel):
    summary: str = Field(description="The abstract of the document")


class Doc(BaseModel):
    url: str
    file_name: str
    content: str
    id: str
    abstract: DocumentSummary


class MindNode(BaseModel):
    id: str = Field(description="mind node uuid")
    doc_list: list[Doc] = Field(description="docs belong to node")
    content: str = Field(description="node content")
    children_ids: list[str] = Field(description="chilren node ids")
    lock: bool = Field(description="lock build")
    deep_level: int = Field(description="max children deep")


# Core Classes
class SummaryExtractor:
    def __init__(self, model):
        self.model = model
        self.agent = Agent(
            name="extract_abstract",
            model=Gemini(model=model), # Instantiate Gemini directly
            description="Extract the abstract from the document",
            instruction="You are a helpful agent who can extract abstracts and summaries from documents. Analyze the provided document content and extract a clear, concise abstract that captures the main points.",
            output_schema=DocumentSummary,
        )
        self.prompt = "Extract the abstract from the document\n"

    async def extract_abstract(self, document: Doc, existing_structure: str = ""):
        """Extract abstract using the updated ADK API pattern"""
        print(f"begin to extract doc:{document.file_name} abstract")
        
        try:
            user_message = self.prompt + document.content[:100] # Use a simplified message for testing
            print(f"User message for {document.file_name}: {user_message}...")
            
            invocation_id = f"inv-{uuid.uuid4()}"
            session_service = InMemorySessionService()
            session = Session(id=f"sess-{uuid.uuid4()}", app_name="dechaos", user_id="user123")
            user_content = types.Content(parts=[types.Part(text=user_message)])
            invocation_context = InvocationContext(agent=self.agent, session=session, invocation_id=invocation_id, user_content=user_content, session_service=session_service)

            response_events = []
            async for event in self.agent.run_async(invocation_context):
                print(f"Received event for {document.file_name}: {event}")
                response_events.append(event)
            
            final_response_content = ""
            for event in response_events:
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            final_response_content += part.text
            response = final_response_content

            print(f"Raw response for {document.file_name}: {response}")
            
            if hasattr(response, 'summary'):
                summary = DocumentSummary(summary=response.summary)
            else:
                summary_text = str(response).strip()
                summary = DocumentSummary(summary=summary_text)
            
            document.abstract = summary
            print(f"end extract doc:{document.file_name} abstract, summary: {summary.summary}")
            return summary
            
        except Exception as e:
            print(f"Error extracting abstract for {document.file_name}: {e}")
            summary = DocumentSummary(summary="")
            document.abstract = summary
            return summary


class DocGroup:
    def __init__(
        self,
        docs,
        outline_extracter,
    ):
        self.source = "path"
        self.docs: list[Doc] = docs
        self.combined_text: str = ""
        self.outline_extracter: SummaryExtractor = outline_extracter

    @classmethod
    def from_docs(self, docs: list[Doc]) -> "DocGroup":
        combined = "\n\n".join(d.content for d in docs)
        return DocGroup(docs=docs, combined_text=combined)

    async def extract_abstract(self) -> list:
        tasks = [self.outline_extracter.extract_abstract(doc) for doc in self.docs]
        results = await asyncio.gather(*tasks)
        print(
            f"group summary extract success! all{len([doc for doc in self.docs if hasattr(doc, 'abstract') and doc.abstract.summary != ''])}"
        )
        return results

    async def constract_outline(self):
        pass


# Helper Functions
def read_single_pdf(path: Path) -> str:
    """同步读取单个 PDF 文本。空则返回空串。"""
    parts = []
    print(f"begin reading pdf: {str(path)}")
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                parts.append(txt)
        print(f"success reading pdf: {str(path)}")

    except Exception as e:
        print(f"[WARN] 读取失败 {path.name}: {e}")
    return "\n\n".join(parts)


async def async_read_single_pdf(path: Path) -> str:
    result = await asyncio.to_thread(read_single_pdf, path)
    return result


async def apply_func(path: Path, cor):
    doc_list = [i for i in path.iterdir() if i.is_file()]
    tasks = [asyncio.create_task(cor(i)) for i in doc_list]
    try:
        response = await asyncio.gather(*tasks)
        return response, None
    except Exception as e:
        return None, e


async def load_docs_from_folder(folder: Path) -> list[Doc]:
    docs: list[Doc] = []
    pdf_list = [pdf_file for pdf_file in sorted(folder.glob("*.pdf"))]
    print("begin read")
    tasks = [async_read_single_pdf(pdf) for pdf in pdf_list]
    text_list = await asyncio.gather(*tasks)
    print("end read")
    for index, text in enumerate(text_list):
        doc = Doc(
            id=pdf_list[index].stem,
            url=str(pdf_list[index]),
            file_name=pdf_list[index].name,
            content=text,
            abstract=DocumentSummary(summary=""),
        )
        docs.append(doc)
    print(f"Loaded {len(docs)} documents: {[d.file_name for d in docs]}")
    return docs


async def generate_structure(docgroup: DocGroup) -> None:
    pass


# Main Execution
async def main(path):
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    print("path:", path)

    # Check GOOGLE_API_KEY
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print(f"GOOGLE_API_KEY is set (first 5 chars): {google_api_key[:5]}...")
    else:
        print("GOOGLE_API_KEY is NOT set.")
        return # Exit if API key is not set

    # Test Gemini directly
    try:
        print("Testing Gemini directly...")
        gemini_instance = Gemini(model=model)
        direct_response = await gemini_instance.generate_content("Hello, what is your name?")
        print(f"Direct Gemini response: {direct_response.text}")
    except Exception as e:
        print(f"Error testing Gemini directly: {e}")
        return # Exit if direct Gemini call fails

    docs = await load_docs_from_folder(Path(path))
    
    extractor = SummaryExtractor(model=model)
    
    docs_group = DocGroup(docs=docs, outline_extracter=extractor)
    
    overall_outline = await docs_group.extract_abstract()
    print(overall_outline)


if __name__ == "__main__":
    asyncio.run(main(r"C:\Users\meidi\Desktop\projects\dechaos\data"))
