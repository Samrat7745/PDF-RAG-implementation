import base64
import mimetypes
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import os

class DiagramDescriber:
    def __init__(self, model_name="llama3.2-vision", prompt =""):
        self.vision_llm = ChatOllama(model=model_name)
        if prompt == "":
            self.prompt = "You will encounter 4 types of images: architectural diagrams, flowcharts, graphs, and screenshots of texts. Describe this diagram in detail. List the components, how they are connected, and the flow of data. If it is a graph, explain the axes, the data points, and any trends or patterns you observe. If it is a flowchart, describe each step and how they are connected. If it is an architectural diagram, explain the components and their relationships. If it is a screenshot of text, ignore it."
        else:
            self.prompt = prompt
    def describe_diagram(self, image_path):
        with open(image_path, "rb") as image_file:
            image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(image_path)
        mime_type = mime_type or "image/png"
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}}
            ]
        )
        response = self.vision_llm.invoke([message])
        return response.content
