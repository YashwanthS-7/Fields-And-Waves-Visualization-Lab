import os
import re
from dotenv import load_dotenv
from pathlib import Path
from google import genai

# -------------------------------------------------
# Load .env
# -------------------------------------------------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# -------------------------------------------------
# Conversation State (IMPORTANT)
# -------------------------------------------------
conversation_states = {}

def get_conversation_state(session_id: str):
    if not session_id:
        session_id = "anonymous"
    if session_id not in conversation_states:
        conversation_states[session_id] = {
            "current_subtopic": None,
            "last_bot_explanation": None,
        }
    return conversation_states[session_id]

FWV_STRUCTURE = {
    "Vector Algebra": [
        "Scalars and Vectors",
        "Vectors",
        "Addition",
        "Multiplication",
        "Triple Product"
    ],
    "Vector Calculus": [
        "Cylindrical Coordinates",
        "Spherical Coordinates",
        "Cartesian, Cylindrical and Spherical",
        "Differential Length, Area and Volume",
        "Del Operator"
    ],
    "Electrostatics": [
        "Intro",
        "Electric Field & Flux",
        "Electric Potential",
        "Electric Dipole"
    ],
    "Maxwell Equations": [
        "Gauss Law",
        "Magnetism",
        "Faraday Law",
        "Ampere Law",
        "Displacement Current",
        "Time Varying Potential",
        "EMF"
    ],
    "Wave Propagation": [
        "Types of Waves",
        "Wave Power",
        "Energy",
        "Plane Wave Analysis",
        "Wave Reflection"
    ],
    
    "Transmission Lines": [
        "Types of Transmission Line",
        "Characteristic Impedance",
        "Smith Chart"
    ]
}

TOPIC_TO_AREA = {
    topic.lower(): area
    for area, topics in FWV_STRUCTURE.items()
    for topic in topics
}

SITE_BASE_URL = os.getenv("FWV_SITE_URL", "https://www.fwvlab.com")

TOPIC_TO_PATH = {
    "scalars and vectors": "/scalars-and-vectors",
    "vectors": "/scalars-and-vectors",
    "addition": "/vector-addition",
    "multiplication": "/vector-multiplication",
    "triple product": "/triple-product",
    "cylindrical coordinates": "/cylindrical-coordinates",
    "spherical coordinates": "/spherical-coordinates",
    "cartesian, cylindrical and spherical": "/cartesian-cylindrical-spherical",
    "differential length, area and volume": "/vector-calculus-intro",
    "del operator": "/del-operator",
    "intro": "/electrostatics-intro",
    "electric field & flux": "/electric-field-and-flux-density",
    "electric potential": "/electric-potential",
    "electric dipole": "/electric-dipole",
    "gauss law": "/gauss-law-contd",
    "magnetism": "/gauss-law-magnetism",
    "faraday law": "/faraday-law",
    "ampere law": "/ampere-law",
    "displacement current": "/displacement-current",
    "time varying potential": "/time-varying-potential",
    "emf": "/transformer-motional-emf",
    "types of waves": "/types-of-waves",
    "wave power": "/wave-power-energy",
    "energy": "/wave-power-energy",
    "plane wave analysis": "/plane-wave-analysis",
    "wave reflection": "/wave-reflection",
    "types of transmission line": "/types-of-transmission-line",
    "characteristic impedance": "/characteristic-impedance",
    "smith chart": "/smith-chart",
}


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def normalize_topic_key(topic: str | None) -> str | None:
    if not topic:
        return None

    normalized = normalize_text(topic)
    normalized = re.sub(r"\s+", " ", normalized)

    alias_map = {
        "faradays law": "faraday law",
        "faraday s law": "faraday law",
        "faraday law": "faraday law",
        "faraday s": "faraday law",
        "smithchart": "smith chart",
        "smith chart": "smith chart",
        "ampere s law": "ampere law",
        "amperes law": "ampere law",
        "displacement current": "displacement current",
    }

    for key, value in alias_map.items():
        if normalized == key or key in normalized:
            return value

    return normalized


def get_topic_link(topic: str) -> str | None:
    topic_key = normalize_topic_key(topic)
    if not topic_key:
        return None

    slug = TOPIC_TO_PATH.get(topic_key)
    if not slug:
        return None

    return f"<a href=\"{SITE_BASE_URL}{slug}\" target=\"_blank\" rel=\"noopener noreferrer\">{topic_key.title()}</a>"


def infer_topic_from_text(text: str) -> str | None:
    if not text:
        return None

    normalized = normalize_text(text)
    normalized = re.sub(r"\s+", " ", normalized)
    topic_key = normalize_topic_key(normalized)
    if topic_key in TOPIC_TO_PATH:
        return topic_key

    for topic in sorted(TOPIC_TO_PATH.keys(), key=len, reverse=True):
        if topic in normalized:
            return topic

    return None


def append_area_and_topic(answer: str, topic: str) -> str:
    area = TOPIC_TO_AREA.get(topic.lower() if isinstance(topic, str) else topic)

    if "To learn more, visit:" in answer:
        answer = re.sub(r"To learn more, visit:.*$", "", answer).strip()

    link = get_topic_link(topic)
    if link:
        if answer and not answer.endswith("."):
            answer = answer + "."
        answer = f"{answer}\n\nTo learn more, visit: {link}"
    elif topic and area:
        answer = f"{answer}\n\nTo learn more, visit: {topic}"

    if not area:
        return answer

    return (
        f"{answer}\n\n"
        f"Area: {area}\n"
        f"Topic: {topic}"
    )


# -------------------------------------------------
# Main Tutor Function
# -------------------------------------------------
def generate_explanation(context: str, question: str, session_id: str = "anonymous") -> str:
    state = get_conversation_state(session_id)
    current_subtopic = state["current_subtopic"]
    last_bot_explanation = state["last_bot_explanation"]

    # -------------------------------------------------
    # SYSTEM INSTRUCTION (THE BRAIN)
    # -------------------------------------------------
    sys_instruct = """
    You are a patient, adaptive teaching assistant for the Fields and Waves Visualisation Lab (FWV Lab).

    YOUR ROLE:
        - You are a tutor. Help students learn concepts step by step.
        - SIZE CONSTRAINT: Provide the full core meaning in exactly ONE or TWO short, complete sentences.
        - COMPLETENESS: Never stop in the middle of a sentence. Finish your thought.

    SOCIAL BEHAVIOR:
        - Respond naturally to greetings (e.g., hi, hello). Do not teach during small talk.

    ACADEMIC TEACHING RULES:
        - Answer ONLY if content exists in the provided FWV Lab context.
        - Stay within the current topic unless the user explicitly changes it.
        - Think internally and NEVER reveal your reasoning. Avoid equations unless asked.

    FOLLOW-UP INTELLIGENCE:
        - For easy/layman requests: simplify the idea in 1-2 sentences.
        - For examples: give one short intuitive example.
        - For word-level doubts: explain the word based on its usage in the last explanation.

    NAVIGATION:
        - End academic responses with: "To learn more, visit: <Current Subtopic>"

    OUTPUT STYLE:
        - Natural teaching tone. No labels, no headings, no bullet points.
    """

    # -------------------------------------------------
    # USER PROMPT (STATE AWARE)
    # -------------------------------------------------
    user_prompt = f"""
    FWV Lab Context:
    {context}

    Current Topic:
    {current_subtopic if current_subtopic else "Not set"}

    Last Explanation Given:
    {last_bot_explanation if last_bot_explanation else "None"}

    User Message:
    {question}

    IMPORTANT:
    - Stay within the current topic.
    - If the user asks for a new topic, update the current topic and teach that.
    - Adapt explanation depth based on the user's intent.
    - Keep it very brief (1-2 sentences).
    - Ensure the explanation has full meaning and does not cut off.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config={
                "system_instruction": sys_instruct,
                "temperature": 0.3,
            }
        )

        if not response.text:
            return (
                "I can help you with Fields and Waves topics from the FWV Lab notes.\n"
                "Please choose a topic from the sidebar to continue."
            )

        final_answer = response.text.strip()
        inferred_topic = infer_topic_from_text(question)

        # -------------------------------------------------
        # Update conversation memory safely
        # -------------------------------------------------
        if inferred_topic:
            state["current_subtopic"] = inferred_topic
        elif "To learn more, visit:" in final_answer:
            state["last_bot_explanation"] = final_answer
            inferred_from_answer = infer_topic_from_text(final_answer.split("visit:")[-1])
            if inferred_from_answer:
                state["current_subtopic"] = inferred_from_answer

        return append_area_and_topic(final_answer, state["current_subtopic"])

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "I'm having trouble accessing the FWV Lab notes right now."

# -------------------------------------------------
# Example Terminal Test
# -------------------------------------------------
if __name__ == "__main__":
    sample_context = """
    Displacement Current:
    Introduced by Maxwell to explain how a changing electric field can produce
    a magnetic field, even in the absence of conduction current.

    Gauss Law:
    The total electric flux through a closed surface is proportional
    to the charge enclosed.

    Faraday Law:
    A changing magnetic field induces an electric field.
    """

    print("FWV AI Tutor (type 'exit' to quit)")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("FWV Bot: Bye! 👋 Keep learning Fields & Waves.")
            break

        reply = generate_explanation(sample_context, user_input)
        print("\nFWV Bot:", reply)
