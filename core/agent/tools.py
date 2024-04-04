from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool, Tool, tool

from core.agent.model import Model

llm = Model.openai_chat_llm()
tools = []

# transcribe tool
# class Transcribe(BaseModel):
#     file: bytes

# def transcribe(file: bytes) -> dict:
#     """Transcribe audio file"""
#     result = transcribe_audio(file)
#     return result

# transcribe_tool = StructuredTool.from_function(
#     func=transcribe,
#     name="transcribe",
#     description="Use when asked to transcribe the audio file.",
#     args_schema=Transcribe,
# )
# tools.append(transcribe_tool)

# class TranscribeResult(BaseModel):
#     result: dict


# def get_full_transcribed_text(result: dict) -> str:
#     """returns the full transcribed text from the result obtained after transcription.

#     Args:
#     - transcribed_text (dict): transcription result

#     Returns:
#     - str: full transcribed text
#     """
#     return result['text']

# tools.append(get_full_transcribed_text)


class Translate(BaseModel):
    text: str
    language: str = "en"


class TranslatedText(BaseModel):
    input_text: str = Field(description="input text to be translated")
    language: str = Field(description="language to be translated to")
    translated_text: str = Field(description="translated text")


class Questions(BaseModel):
    question: List[str] = Field(description="List of questions")


class CreateQuestions(BaseModel):
    num_of_questions: int
    text: str


def build_tools():
    def translate(text: str, language: str = "en") -> str:
        """Translates the text to the specified language"""
        parser = PydanticOutputParser(pydantic_object=TranslatedText)
        prompt = PromptTemplate(
            template="Translate the following {text} to {language} \n{format_instructions}\n",
            input_variables=["text", "language"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser
        result = chain.invoke({"text": text, "language": language})
        return result

    translate_tool = StructuredTool.from_function(
        func=translate,
        name="translate",
        description="Use when asked to translate the text to the specified language. Return only translated text.",
        args_schema=Translate,
    )

    tools.append(translate_tool)

    ## Question creation tool

    def create_questions(text: str, num_of_questions: int):
        template = """You are an expert question generator bot. Given the following context, create {num_of_questions} question.\nContext:{text}\n\n{format_instructions}"""
        parser = PydanticOutputParser(pydantic_object=Questions)

        prompt = PromptTemplate(
            template=template,
            input_variables=["num_of_questions", "text"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | llm | parser

        return chain.invoke({"num_of_questions": num_of_questions, "text": text})

    create_questions_from_context_tool = StructuredTool.from_function(
        name="create_questions_from_context",
        func=create_questions,
        description="Use when asked to create questions based on the context.\nArgs:\n- text (str): input text\n- num of question (int): number of questions to be created\nReturns:\n- str: generated questions",
        args_schema=CreateQuestions,
    )
    tools.append(create_questions_from_context_tool)

    return tools
