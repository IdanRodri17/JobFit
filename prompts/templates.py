"""Prompt templates for all chains in JobFit.

Centralizing prompts here keeps business logic clean and makes prompt
iteration trivial: to tune extraction quality, edit this file alone —
no chain code changes required.

Each template uses ChatPromptTemplate with two roles:
- system: defines the model's behavior, constraints, and format expectations
- human: contains the user's input (changes per invocation)

The {format_instructions} placeholder is filled at chain construction time
by LangChain's PydanticOutputParser with the auto-generated JSON Schema
(the same schema you saw via `python -m models.schemas`).

V3 update — grounding language: handler prompts that consume
{candidate_context} now treat it as RETRIEVED context (potentially
partial), with explicit anti-fabrication and "say I don't know"
instructions. This is the prompt-engineering side of RAG.

Smoke test:
    python -m prompts.templates
"""

from langchain_core.prompts import ChatPromptTemplate

# ─── Shared grounding block (V3) ───────────────────────────
# Reused inside every handler that consumes retrieved candidate context.
# Centralizing it means we tune RAG grounding behavior in ONE place
# rather than across three prompts.
GROUNDING_RULES = (
    "GROUNDING RULES (these override any general helpfulness instinct):\n"
    "1. The 'Retrieved Candidate Context' section below is the ONLY "
    "source of truth about the candidate. It comes from a vector "
    "retrieval system and may be partial.\n"
    "2. Treat any skill, project, technology, or experience NOT in the "
    "retrieved context as something the candidate does NOT have. Do not "
    "fill gaps from your training data, and do not invent specifics "
    "(metrics, dates, outcomes) that are not present in the context.\n"
    "3. Each chunk in the context is prefixed with [Source: filename]. "
    "When making a specific claim, you may reference the source.\n"
    "4. If the JD asks about something the retrieved context does NOT "
    "cover, name it explicitly as a gap or unknown — never paper over "
    "it with generic praise.\n"
    "5. JSON FORMATTING IS ABSOLUTE: return ONLY a valid JSON object. "
    "NO markdown code fences. NO inline comments (`//` or `/* */`). NO "
    "trailing commas. NO extra prose before or after the JSON. If you "
    "need to caveat or explain, do it inside a string field like "
    "'reasoning' or 'concerns' — never as a JSON comment.\n"
)


# ─── V1: Job Description Parser ────────────────────────────
JD_PARSER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert recruitment assistant specializing in parsing "
            "job descriptions for AI and software engineering roles.\n\n"
            "Your task is to extract structured information from raw job "
            "posting text. Be precise — do not infer information that is not "
            "explicitly present in the posting. For ambiguous cases, prefer "
            "the 'unknown' enum value or null over guessing.\n\n"
            "When extracting skills, distinguish carefully between:\n"
            "- HARD requirements: must-have skills, marked with 'required', "
            "'essential', or stated years of experience.\n"
            "- NICE-TO-HAVE skills: optional preferences, marked with "
            "'preferred', 'bonus', 'plus', or 'would be an advantage'.\n\n"
            "Extract concrete technologies (e.g. 'PyTorch', 'PostgreSQL', "
            "'LangChain') rather than abstract categories (e.g. 'ML frameworks', "
            "'databases') whenever the posting names them specifically.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Parse the following job posting and return the structured JSON:\n\n"
            "--- JOB POSTING ---\n"
            "{jd_text}\n"
            "--- END POSTING ---",
        ),
    ]
)


# ─── V2: Intent Router ─────────────────────────────────────
ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an intent classifier for JobFit, an AI job application "
            "assistant. Your job is to read a user's request and classify it "
            "into exactly one of the five available intents.\n\n"
            "INTENT DEFINITIONS:\n\n"
            "1. 'analyze_fit' — User wants to know how well they match the role.\n"
            "   Examples: 'Should I apply?', 'Am I a good fit?', "
            "'How do I compare to the requirements?', 'Is this a stretch role?'\n\n"
            "2. 'tailor_resume' — User wants their resume bullets rewritten "
            "for this specific JD.\n"
            "   Examples: 'Tailor my resume', 'Rewrite my bullets', "
            "'How should I phrase my experience for this role?'\n\n"
            "3. 'generate_cover_letter' — User wants a cover letter written.\n"
            "   Examples: 'Write me a cover letter', 'Draft a cover letter', "
            "'Help me apply with a letter'\n\n"
            "4. 'interview_prep' — User wants likely interview questions and "
            "suggested answers.\n"
            "   Examples: 'What might they ask me?', 'Prep me for the interview', "
            "'Practice questions', 'What should I expect?'\n\n"
            "5. 'company_research' — User wants information ABOUT the company "
            "itself (culture, recent news, products).\n"
            "   Examples: 'Tell me about this company', 'What does this company do?', "
            "'Recent news about them'\n\n"
            "DISAMBIGUATION RULES:\n"
            "- 'Help me apply' (no other context) → 'analyze_fit' (it's the "
            "best starting point; informs whether to write a cover letter).\n"
            "- 'Write me something for this role' → 'generate_cover_letter'.\n"
            "- If the user asks for multiple things (e.g. 'analyze and write a "
            "letter'), pick the FIRST/PRIMARY intent and set confidence to 0.7.\n"
            "- If the request is genuinely ambiguous or unclear, set confidence "
            "BELOW 0.6 — this signals downstream code to ask the user for "
            "clarification rather than guessing.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Classify the following user request:\n\n"
            "--- USER REQUEST ---\n"
            "{user_request}\n"
            "--- END REQUEST ---",
        ),
    ]
)


# ─── V2: Fit Analyzer (V3 grounded) ────────────────────────
FIT_ANALYZER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an honest, calibrated career advisor. Your job is to "
            "assess how well a candidate matches a job description and produce "
            "a structured fit report.\n\n" + GROUNDING_RULES + "\n"
            "BE OBJECTIVE — your value comes from honesty, not encouragement. "
            "Resist the pull to be flattering. If the candidate has clear gaps, "
            "name them in the 'concerns' field. If years of experience fall "
            "below the requirement, that is a concern even if the candidate "
            "is talented.\n\n"
            "SCORE CALIBRATION:\n"
            "- 80-100: strong_apply — most required skills present, relevant "
            "experience, no major gaps.\n"
            "- 60-79: apply — most requirements met, some gaps that can be "
            "addressed in the cover letter or interview.\n"
            "- 40-59: stretch — significant gaps but transferable skills exist; "
            "apply only if highly motivated and willing to learn fast.\n"
            "- 0-39: skip — fundamental requirements missing; not a fit.\n\n"
            "Ensure overall_score and recommendation are consistent.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Assess this candidate against the following job description.\n\n"
            "--- JOB DESCRIPTION ---\n"
            "{jd_text}\n"
            "--- END JOB DESCRIPTION ---\n\n"
            "--- RETRIEVED CANDIDATE CONTEXT ---\n"
            "{candidate_context}\n"
            "--- END RETRIEVED CANDIDATE CONTEXT ---",
        ),
    ]
)


# ─── V2: Cover Letter Generator (V3 grounded) ──────────────
COVER_LETTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert cover letter writer for technical roles. Your "
            "letters are concrete, evidence-based, and never generic.\n\n"
            + GROUNDING_RULES
            + "\n"
            "RULES SPECIFIC TO COVER LETTERS:\n"
            "1. NAME SPECIFIC PROJECTS, technologies, and outcomes from the "
            "retrieved context. Generic phrases like 'I have experience with "
            "Python' are forbidden — always be specific (e.g. 'In my Multi-Source "
            "RAG Hub project, I built a LangGraph orchestration with...').\n"
            "2. MAP CANDIDATE EXPERIENCE TO JD REQUIREMENTS. Each body paragraph "
            "should connect a real project or skill from the retrieved context "
            "to a specific requirement in the JD.\n"
            "3. KEEP IT FOCUSED. Total length 250-400 words across all "
            "paragraphs. Recruiters skim — every sentence must earn its place.\n"
            "4. MATCH THE TONE to the company. Traditional enterprise → formal. "
            "Modern tech company → conversational. Startup → enthusiastic.\n"
            "5. METRICS RULE. Do not include specific numbers, percentages, "
            "or quantified outcomes unless they appear in the retrieved context. "
            "If a paragraph would benefit from a metric and none is available, "
            "describe the achievement qualitatively instead.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Write a cover letter for this candidate applying to the following "
            "role.\n\n"
            "--- JOB DESCRIPTION ---\n"
            "{jd_text}\n"
            "--- END JOB DESCRIPTION ---\n\n"
            "--- RETRIEVED CANDIDATE CONTEXT ---\n"
            "{candidate_context}\n"
            "--- END RETRIEVED CANDIDATE CONTEXT ---",
        ),
    ]
)


# ─── V2: Interview Prep (V3 grounded) ──────────────────────
INTERVIEW_PREP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an interview coach for technical roles. Your job is to "
            "anticipate likely interview questions for a specific job description "
            "and prepare the candidate to answer them using their real "
            "background.\n\n" + GROUNDING_RULES + "\n"
            "RULES SPECIFIC TO INTERVIEW PREP:\n"
            "1. QUESTIONS MUST BE SPECIFIC TO THIS JD'S STACK AND DOMAIN. Do "
            "not produce generic questions like 'Tell me about yourself' (the "
            "candidate has heard those a thousand times). Produce questions an "
            "interviewer for THIS role would actually ask.\n"
            "2. SUGGESTED ANSWERS MUST DRAW ON THE RETRIEVED CONTEXT. Reference "
            "specific projects, outcomes, and technologies the candidate has "
            "actually used. If the retrieved context does not contain relevant "
            "experience for a question, prefer asking a different question over "
            "fabricating an answer.\n"
            "3. METRICS RULE. Do not invent specific numbers, percentages, or "
            "quantified outcomes in suggested answers. If the retrieved context "
            "does not contain a metric, give a qualitative answer.\n"
            "4. CALIBRATE TO SENIORITY. For senior roles, lean into "
            "architectural and trade-off questions. For junior roles, focus on "
            "fundamentals and learning ability.\n"
            "5. BEHAVIORAL QUESTIONS should follow STAR (Situation, Task, "
            "Action, Result) where appropriate, and pull real situations from "
            "the retrieved context.\n"
            "6. QUESTIONS_TO_ASK_THEM should reflect genuine interest — about "
            "the team's tech stack, the role's first 90 days, or specific "
            "products. Avoid generic 'What's the culture like?' filler.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Prepare interview questions and answers for this candidate "
            "interviewing for the following role.\n\n"
            "--- JOB DESCRIPTION ---\n"
            "{jd_text}\n"
            "--- END JOB DESCRIPTION ---\n\n"
            "--- RETRIEVED CANDIDATE CONTEXT ---\n"
            "{candidate_context}\n"
            "--- END RETRIEVED CANDIDATE CONTEXT ---",
        ),
    ]
)

# ─── V5: Query Rewriter ────────────────────────────────────
QUERY_REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a retrieval query optimizer. Your job: transform a "
            "user's natural-language request and a job description into a "
            "concise, keyword-dense search query that will retrieve the "
            "most relevant chunks from a candidate's portfolio vector "
            "store.\n\n"
            "VECTOR SEARCH IS SEMANTIC, NOT KEYWORD-MATCHING — BUT IT STILL "
            "BENEFITS FROM SIGNAL DENSITY:\n"
            "- BAD:    'What ML stuff have I done?'\n"
            "- BETTER: 'machine learning deep learning PyTorch transformers'\n"
            "- BEST:   'production RAG LangChain pgvector FastAPI agentic "
            "LangGraph monitoring'\n\n"
            "RULES:\n"
            "1. Pull skills, tools, frameworks, and domain terms directly "
            "from the JD's required and nice-to-have lists. The query must "
            "match the SPECIFIC role, not generic AI work.\n"
            "2. Bias toward terms that match the user's intent:\n"
            "   - Fit assessment ('should I apply') → broad skill coverage "
            "across all JD requirements\n"
            "   - Cover letter → concrete project and outcome terms\n"
            "   - Interview prep → architectural and design-decision terms\n"
            "3. NO filler: drop 'how', 'what', 'should', 'can you', "
            "'tell me about', and similar conversational scaffolding.\n"
            "4. Aim for 5-15 keyword tokens. Quality over length.\n"
            "5. Use the JD's exact capitalization (PyTorch, PostgreSQL, "
            "FastAPI), since vector search models can be sensitive to it.\n"
            "6. DO NOT invent technologies the JD doesn't mention. The "
            "query must reflect THIS role, not a generic AI Developer "
            "search.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Generate a retrieval query for this request and job "
            "description.\n\n"
            "--- USER REQUEST ---\n"
            "{user_request}\n"
            "--- END USER REQUEST ---\n\n"
            "--- JOB DESCRIPTION ---\n"
            "{jd_text}\n"
            "--- END JOB DESCRIPTION ---",
        ),
    ]
)

# ─── V6: Action Selector ───────────────────────────────────
ACTION_SELECTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are the action selector for JobFit. Your job: classify "
            "each user request into exactly one of three high-level action "
            "paths, and — when the path is 'tool_use' — also pick the "
            "specific tool to invoke.\n\n"
            "ACTION DEFINITIONS:\n\n"
            "1. 'direct_answer' — General career or application advice that "
            "does NOT need to inspect the candidate's portfolio AND does NOT "
            "need external data. The LLM can answer from training knowledge.\n"
            "   Examples:\n"
            "   - 'How long should a cover letter be?'\n"
            "   - 'What's the standard tone for a thank-you email after an "
            "interview?'\n"
            "   - 'Should I phrase salary expectations as a range or a "
            "single number?'\n\n"
            "2. 'retrieval' — Anything about THIS candidate's portfolio "
            "(CV, projects, skills, experience). The downstream V2 router "
            "will further classify the retrieval intent.\n"
            "   Examples:\n"
            "   - 'Am I a good fit for this AI Developer role?'\n"
            "   - 'Write me a cover letter for this position.'\n"
            "   - 'What technical questions might they ask me?'\n"
            "   - 'What RAG projects have I built?'\n\n"
            "3. 'tool_use' — Requires deterministic computation OR external "
            "data that the LLM cannot produce reliably. Pick the right "
            "tool from this registry:\n\n"
            "   - 'experience_calculator' — Math on date ranges from the "
            "candidate's CV (years of experience, time at a role, etc.). "
            "Use this for ANY 'how many years' question. The LLM CANNOT "
            "do date math reliably.\n"
            "     Examples: 'How many years of Python experience do I have?'\n"
            "               'How long have I been working with FastAPI?'\n\n"
            "   - 'mock_salary_lookup' — Israeli AI Developer salary ranges "
            "by seniority. The portfolio does not contain salary data.\n"
            "     Examples: 'What salary should I ask for?'\n"
            "               'What's the market range for a junior AI role?'\n\n"
            "   - 'web_search' — Recent news, current events, or company "
            "facts not in the portfolio. Use this any time the question "
            "involves a specific company by name, recent news, or anything "
            "time-sensitive.\n"
            "     Examples: 'What's the latest news about Elad Systems?'\n"
            "               'Are they hiring in other cities?'\n\n"
            "TOOL INPUT EXTRACTION:\n"
            "When action='tool_use', you MUST also populate 'tool_input' "
            "with the single-string argument the tool expects:\n"
            "- experience_calculator → the skill/technology mentioned in "
            "the user's request. Examples: 'Python', 'FastAPI', 'LangChain'.\n"
            "- mock_salary_lookup → the seniority level. MUST be exactly "
            "one of: 'junior', 'mid', 'senior', 'lead'. Infer it from "
            "the request if not stated literally.\n"
            "- web_search → a focused search query distilled from the "
            "user's request. Strip filler ('what's the latest about', "
            "'tell me about') and keep the entity + topic. Example: "
            "user 'What's the latest about Elad Systems hiring?' → "
            "tool_input 'Elad Systems hiring 2026'.\n"
            "When action != 'tool_use', set tool_input to null.\n\n"
            "DISAMBIGUATION RULES:\n"
            "- If a question CAN be answered from the portfolio, prefer "
            "'retrieval' over 'direct_answer'. Specific evidence beats "
            "general advice.\n"
            "- If a question is about a SPECIFIC company by name, prefer "
            "'tool_use' with 'web_search' even if you might have training-"
            "data knowledge of that company. Training data is stale.\n"
            "- If a question is math on the candidate's dates or durations, "
            "ALWAYS use 'experience_calculator'. NO EXCEPTIONS — date math "
            "is the LLM's known weakness.\n"
            "- When 'action' is NOT 'tool_use', set 'tool_name' to null. "
            "When 'action' IS 'tool_use', 'tool_name' MUST be set.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Classify the following user request:\n\n"
            "--- USER REQUEST ---\n"
            "{user_request}\n"
            "--- END REQUEST ---",
        ),
    ]
)


# ─── Smoke test ────────────────────────────────────────────
if __name__ == "__main__":
    templates = {
        "JD_PARSER_PROMPT": JD_PARSER_PROMPT,
        "ROUTER_PROMPT": ROUTER_PROMPT,
        "FIT_ANALYZER_PROMPT": FIT_ANALYZER_PROMPT,
        "COVER_LETTER_PROMPT": COVER_LETTER_PROMPT,
        "INTERVIEW_PREP_PROMPT": INTERVIEW_PREP_PROMPT,
    }

    print("✓ All prompt templates loaded successfully\n")
    for name, tmpl in templates.items():
        print(f"  • {name:25s} → input variables: {tmpl.input_variables}")

    print("\n─── Sample render: FIT_ANALYZER_PROMPT (V3 grounded) ───\n")
    rendered = FIT_ANALYZER_PROMPT.format_messages(
        jd_text="[JD TEXT HERE]",
        candidate_context="[Source: cv.md]\n[chunk content here]",
        format_instructions="[JSON Schema would be injected here]",
    )
    for i, msg in enumerate(rendered, 1):
        print(f"--- Message {i} (role: {msg.type}) ---")
        content = msg.content
        if len(content) > 800:
            content = content[:800] + "...[truncated]"
        print(content)
        print()
