from __future__ import annotations

from textwrap import dedent

from shiny import App, ui, reactive
from dotenv import load_dotenv
from chatlas import ChatOpenAI

from rag import get_top_k_similar_documents  

load_dotenv()  


system_prompt = dedent("""
    You are a course tutor for INFO 4940 / 5940 at Cornell.

    Your role:
    - Answer questions about topics covered in this course (LLMs, prompts, RAG, evaluation, dashboards, etc.).
    - Help students reason through homework and projects WITHOUT giving full solutions.
    - Explain code and concepts clearly using R or Python examples when requested.
    - Use the provided context (syllabus, assignment text, project description) as your primary factual source.

    What you MUST do:
    - Be honest about what you know and what you don't.
    - Encourage good academic integrity: guide, don't solve.
    - Prefer course materials for questions about policies, grading, and due dates.
    - When giving code, keep it fairly short and explain it.

    What you MUST NOT do:
    - Do NOT write full homework or project answers that could be submitted as-is.
    - Do NOT fabricate course policies or due dates.
    - Do NOT provide long, copy-pasteable solutions to graded questions.

    Style:
    - Friendly, concise, and concrete.
    - If the student asks for a specific language, use that (R or Python).
    - If they don't specify, default to Python for code.
    - If their question is vague, ask one clarifying question before answering.
""").strip()
#LLM

chat_client = ChatOpenAI(
    model="gpt-4.1-mini",  # adjust to your allowed model
    system_prompt=system_prompt,
)

# UI

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Tutor settings"),

        ui.input_radio_buttons(
            "help_mode",
            "What do you need help with?",
            choices=[
                "Concepts (lectures/readings)",
                "Assignments & projects",
                "Code debugging",
                "Course policies (syllabus)",
                "Study strategies",
            ],
            selected="Concepts (lectures/readings)",
        ),

        ui.input_radio_buttons(
            "pref_lang",
            "Preferred language for examples",
            choices=["No preference", "Python", "R"],
            selected="No preference",
        ),

        ui.hr(),
        ui.input_action_button(
            "show_tips",
            "Show tips for good questions",
            class_="btn-secondary w-100",
        ),
        ui.br(),
        ui.input_action_button(
            "show_examples",
            "Show example questions",
            class_="btn-outline-secondary w-100",
        ),
    ),

    ui.layout_columns(
        ui.card(
            ui.card_header("INFO 4940 / 5940 Tutor Bot"),
            ui.markdown(
                "Ask about **lectures**, **homework**, **projects**, "
                "**course policies**, or **how to structure your code**.\n\n"
            ),
            ui.chat_ui(
                "tutor_chat",
                placeholder="Ask your question here…",
            ),
        ),
    ),

    title="INFO 4940 / 5940 Tutor Bot",
    fillable=True,
    fillable_mobile=True,
)

# Server

def server(input, output, session):

    chat = ui.Chat(id="tutor_chat")

    # Popups

    tips_modal = ui.modal(
        ui.h4("Tips for using the tutor"),
        ui.tags.ul(
            ui.tags.li("Ask focused questions (e.g., one assignment or concept at a time)."),
            ui.tags.li("Mention the assignment number or topic when relevant."),
            ui.tags.li("If you're stuck on code, paste the error message and a small code snippet."),
        ),
        title="Tips",
        easy_close=True,
        footer=ui.input_action_button(
            "close_tips",
            "Close",
            class_="btn-primary"
        ),
    )

    examples_modal = ui.modal(
        ui.h4("Example questions"),
        ui.tags.ul(
            ui.tags.li("“What is RAG and how is it different from fine-tuning?”"),
            ui.tags.li("“Can you remind me of the evaluation criteria for Homework 4?”"),
            ui.tags.li("“How should I structure a system prompt for a labeling task?”"),
            ui.tags.li("“What is the late policy for homework in this class?”"),
        ),
        title="Example questions",
        easy_close=True,
        footer=ui.input_action_button(
            "close_examples",
            "Close",
            class_="btn-primary"
        ),
    )

    @reactive.Effect
    @reactive.event(input.show_tips)
    def _show_tips():
        ui.modal_show(tips_modal)

    @reactive.Effect
    @reactive.event(input.close_tips)
    def _close_tips():
        ui.modal_remove()

    @reactive.Effect
    @reactive.event(input.show_examples)
    def _show_examples():
        ui.modal_show(examples_modal)

    @reactive.Effect
    @reactive.event(input.close_examples)
    def _close_examples():
        ui.modal_remove()

    # Chat handler

    @chat.on_user_submit
    async def handle_user_input(user_input: str):
        help_mode = input.help_mode()
        pref_lang = input.pref_lang()

        # 1. Retrieve relevant course docs via RAG
        top_docs = get_top_k_similar_documents(user_input, top_k=4)

        if top_docs:
            context_block = "\n\n---\n\n".join(top_docs)
        else:
            context_block = "No course documents were retrieved."

        # 2. Build full prompt
        prompt = f"""You are tutoring an INFO 4940 / 5940 student.

Student help mode: {help_mode}
Preferred language for examples: {pref_lang}

Use the context below, if relevant, to answer their question.
If the context does not answer the question, say that and answer
based on general LLM knowledge, but do not make up specific course
policies or due dates.

Context:
{context_block}

Student question:
{user_input}
"""

        # 3. Stream back to chat UI
        response_stream = await chat_client.stream_async(prompt)
        await chat.append_message_stream(response_stream)


app = App(app_ui, server)
