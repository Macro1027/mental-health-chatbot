# Import the FewShotPromptTemplate class from langchain module
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate

# Define examples that include user queries and AI's answers specific to Kaggle competitions
examples = [
    {
        "query": "How do I start with Kaggle competitions?",
        "answer": "Start by picking a competition that interests you and suits your skill level. Don't worry about winning; focus on learning and improving your skills."
    },
    {
        "query": "What should I do if my model isn't performing well?",
        "answer": "It's all part of the process! Try exploring different models, tuning your hyperparameters, and don't forget to check the forums for tips from other Kagglers."
    },
    {
        "query": "How can I find a team to join on Kaggle?",
        "answer": "Check out the competition's discussion forums. Many teams look for members there, or you can post your own interest in joining a team. It's a great way to learn from others and share your skills."
    }
]

# Define the format for how each example should be presented in the prompt
example_template = """
User: {query}
AI: {answer}
"""

# Create an instance of PromptTemplate for formatting the examples
example_prompt = PromptTemplate(
    input_variables=['query', 'answer'],
    template=example_template
)

# Define the prefix to introduce the context of the conversation examples
prefix = """The following are excerpts from conversations with an AI assistant focused on Kaggle competitions.
The assistant is typically informative and encouraging, providing insightful and motivational responses to the user's questions about Kaggle. Here are some examples:
"""

# Define the suffix that specifies the format for presenting the new query to the AI
suffix = """
User: {query}
AI: """

# Create an instance of FewShotPromptTemplate with the defined examples, templates, and formatting
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

query = "Is participating in Kaggle competitions worth my time?"

print(few_shot_prompt_template.format(query=query))