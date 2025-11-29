from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. Define the Standard Output Format for all Experts
class EditSuggestion(BaseModel):
    original_text: str = Field(..., description="The specific snippet being flagged")
    suggestion: str = Field(..., description="The corrected version")
    reasoning: str = Field(..., description="Why this change is required")
    confidence: int = Field(..., description="1-10 scale of necessity")

class ExpertOutput(BaseModel):
    suggestions: List[EditSuggestion]

# 2. Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(ExpertOutput)

# 3. Create Expert Prompts
# Expert A: Syntax
syntax_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict Syntax Expert. Fix only grammatical, spelling, and punctuation errors. Ignore style."),
    ("user", "Analyze this text: {text}")
])

# Expert B: Style
style_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Executive Editor. Focus on clarity, removing passive voice, and ensuring professional tone."),
    ("user", "Analyze this text: {text}")
])

# 4. The 'Chief Editor' (Aggregator)
# This uses the raw text + the suggestions to rewrite the final version
aggregator_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are the Chief Editor. You have received suggestions from multiple sub-editors."),
    ("user", """
    Original Text: {original_text}
    
    Syntax Suggestions: {syntax_data}
    Style Suggestions: {style_data}
    
    Task: Rewrite the original text incorporating valid suggestions. 
    Prioritize Syntax correctness over Style. If suggestions conflict, use your best judgment.
    """)
])

# 5. The Chain Execution (Pseudo-code for the flow)
def run_moe_pipeline(text_chunk):
    # Run Experts in Parallel
    chain_syntax = syntax_prompt | structured_llm
    chain_style = style_prompt | structured_llm
    
    # Execute
    syntax_result = chain_syntax.invoke({"text": text_chunk})
    style_result = chain_style.invoke({"text": text_chunk})
    
    # Aggregation
    final_chain = aggregator_prompt | llm
    final_doc = final_chain.invoke({
        "original_text": text_chunk,
        "syntax_data": syntax_result,
        "style_data": style_result
    })
    
    return final_doc.content