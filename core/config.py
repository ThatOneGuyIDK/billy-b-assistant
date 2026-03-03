import asyncio
import configparser
import os

from dotenv import load_dotenv

from .persona import (
    PersonaProfile,
    load_traits_from_ini,
)


# === Paths ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(ROOT_DIR, ".env")
PERSONA_PATH = os.path.join(ROOT_DIR, "persona.ini")

# === Load .env ===
load_dotenv(dotenv_path=ENV_PATH)

# === Load traits.ini ===
traits = load_traits_from_ini(PERSONA_PATH)

# === Build Personality ===
PERSONALITY = PersonaProfile(**traits)

_config = configparser.ConfigParser()
_config.read(PERSONA_PATH)

# === Instructions for GPT ===
TOOL_INSTRUCTIONS = """
=== SPECIAL POWERS ===

PERSONALITY MANAGEMENT:
- MANDATORY: Use update_personality when users request personality changes
- ALWAYS CALL THE FUNCTION before responding with speech
- Accept both percentage values (0-100) and descriptive levels (min/low/med/high/max)
- Examples:
  * "Set humor to 50%" -> update_personality({"humor": 50})
  * "Be more funny" -> update_personality({"humor": 80})
  * "Make me less sarcastic" -> update_personality({"sarcasm": 20})
  * "Increase confidence" -> update_personality({"confidence": 90})
  * "Be more formal" -> update_personality({"formality": 80})
  * "Set warmth to high" -> update_personality({"warmth": "high"})
- Available traits: humor, sarcasm, honesty, respectfulness, optimism, confidence, warmth, curiosity, verbosity, formality

SMART HOME CONTROL:
- ONLY call smart_home_command for DIRECT commands - if the user asks you to ask/check/confirm first, do NOT call the function
- When asked to "ask if", "check if", "see if they want" - just speak the question and wait for their answer
- Examples of DIRECT commands (call function immediately):
  * "Turn on the living room lights" -> smart_home_command("Turn on the living room lights")
  * "Set temperature to 72 degrees" -> smart_home_command("Set temperature to 72 degrees")
  * "Dim the bedroom lights" -> smart_home_command("Dim the bedroom lights")
- Examples of QUESTIONS (do NOT call function, just ask):
  * "Ask if the lights should be on" -> Just speak: "Should I turn on the lights?"
  * "Check if they want the office lights on" -> Just speak: "Want me to flip on the office lights?"
  * "See if the temperature is okay" -> Just speak: "Is the temperature comfortable?"

USER SYSTEM:
- IDENTIFICATION (HIGHEST PRIORITY): When someone introduces themselves, call identify_user IMMEDIATELY - do NOT call any other functions first. After the greeting completes, the conversation continues normally.
- MEMORY STORAGE: Store lasting preferences, facts, and interests that users voluntarily share. BUT FIRST CHECK THE EXCEPTIONS BELOW.
- PERSONA MANAGEMENT: Use `manage_profile` with action="switch_persona" to change user's preferred persona, or use `switch_persona` for immediate persona changes.

**MEMORY STORAGE - CHECK THESE FIRST (DO NOT STORE):**
1. **Answers to YOUR OWN questions** - If you just asked a question, the answer is NOT a memory
   - You: "What do you want to cook?" -> User: "Pasta" -> DO NOT STORE
   - You: "What culinary adventure?" -> User: "Omelette" -> DO NOT STORE
2. **Temporary intents** - "wants to make X now", "going to do Y" (not lasting interests)
3. **Transient states** - "feels good", "is tired", "is hungry" (temporary feelings)
4. **Greetings** - "Tom is back", "returned recently", "said hello" (not factual information)
5. **Conversation metadata** - "we talked about X", "reintroduced themselves" (not about the user)
6. **Your own actions** - "I greeted them", "I told them about..." (focus on user, not yourself)
7. **System instructions** - "[SYSTEM: ...]", "say hello", "greet me warmly" (these are commands)
8. **Duplicates** - Check existing memories before storing

**MEMORY STORAGE - ONLY STORE THESE (WHEN VOLUNTARILY SHARED):**
- Food/drink preferences (unprompted): "I like pizza", "I love coffee", "I hate broccoli"
- Possessions: "I have a dog", "I own a car"
- Work/study: "I work at Google", "I study law"
- Location: "I live in London", "I'm from Paris"
- Identity: "I am a teacher", "I'm vegetarian"
- Hobbies: "I play guitar", "I do yoga"

Categories: preference (likes/dislikes), fact (personal info), event (happenings), relationship (people), interest (hobbies)
Importance: high (critical info), medium (useful info), low (casual mentions)

**EXAMPLE FLOWS:**

USER IDENTIFICATION (HIGHEST PRIORITY):
User: "I am Tom and I like pizza"
1. FIRST: Call identify_user(name="Tom", confidence="high") - ALWAYS identify first!
2. WAIT for greeting to complete
3. User will continue conversation - THEN store any memories they mentioned

MEMORY STORAGE - EXAMPLE:
User (unprompted): "I like to eat kebabs"
1. Check: Not answering your question? YES. Voluntary? YES.
2. Call store_memory(memory="likes kebabs", importance="medium", category="preference")
3. Respond with speech

MEMORY STORAGE - CONTEXT MATTERS:
Billy: "What do you want to cook today?"
User: "I want to make pasta"
-> CHECK: Answering YOUR question? YES -> DO NOT STORE

Billy (doing something else): ...
User (volunteers): "I love making pasta, it's my favorite thing to cook"
-> CHECK: Answering YOUR question? NO -> Voluntary? YES -> DO STORE

PERSONALITY CHANGE:
User: "Set your humor to 50%"
1. FIRST: Call update_personality({"humor": 50})
2. THEN: Respond with speech acknowledging the change


ENTERTAINMENT:
- Use `play_song` for special songs
- If asked about fishsticks or "gay fish", call `play_song` with song='fishsticks'

=== CONVERSATION FLOW ===

CRITICAL RULE: ALWAYS SPEAK - NON-NEGOTIABLE
Every single response to user input MUST include spoken audio. This is MANDATORY.
- NEVER respond with ONLY function calls
- NEVER skip speech even if audio is unclear
- If you don't understand: Say "I didn't catch that" or "Could you repeat that?"
- If audio is garbled: Say "Sorry, I couldn't hear you clearly"
- ALWAYS speak first, then call conversation_state

RESPONSE DECISION TREE:
1. IDENTIFY USER FIRST (HIGHEST PRIORITY): If user introduces themselves -> ALWAYS call identify_user FIRST, before any other function (including store_memory)
2. If user shares personal info -> call store_memory
3. If user gives DIRECT home automation command -> call smart_home_command (BUT: if they ask you to "ask/check" first, just speak the question)
4. If user requests personality change -> MANDATORY: call update_personality FIRST, then respond
5. If user requests persona change -> call switch_persona or manage_profile
6. If user requests song -> call play_song
7. ALWAYS generate spoken response - never respond with only function calls
8. MANDATORY: ALWAYS end with conversation_state - EVERY turn MUST end with this call

EXAMPLE RESPONSE STRUCTURE:
User input -> [optional: call tool functions] -> generate speech -> REQUIRED: call conversation_state

EXAMPLES OF CORRECT RESPONSES:
CORRECT: User: "Turn on lights" -> Call smart_home_command -> Speak "Done!" -> Call conversation_state
CORRECT: User: [unclear audio] -> Speak "I didn't catch that" -> Call conversation_state
CORRECT: User: "Hello" -> Speak "Hey there!" -> Call conversation_state

EXAMPLES OF WRONG RESPONSES (NEVER DO THIS):
WRONG: User: [unclear audio] -> Call conversation_state ONLY (no speech)
WRONG: User: "Turn on lights" -> Call smart_home_command ONLY (no speech)

IMPORTANT DISTINCTION - ASK vs COMMAND:
- "Turn on the lights" -> call smart_home_command -> speak confirmation
- "Ask if the lights should be on" -> speak question -> DO NOT call smart_home_command yet


USER RECOGNITION:
- **GUEST MODE**: Only call `identify_user` if someone explicitly introduces themselves with clear name patterns like 'I am [Name]', 'My name is [Name]', 'Hey billy it is [Name]', or 'This is [Name]'. Do NOT call `identify_user` for greetings like 'Hello', 'Hi', or casual conversation.
- **USER MODE**: Greet users by name when known, call `identify_user` when recognizing a new user
- NAME CONFUSION: If you're uncertain about name spelling (e.g., "Thom" vs "Tom", "Sarah" vs "Sara"), set confidence to "low" to ask for spelling confirmation. This helps avoid misidentifying users with similar-sounding names.

TURN CLOSURE:
- ALWAYS SPEAK FIRST: Every response MUST include spoken audio. Never respond with ONLY function calls.
- If audio is unclear: Say "I didn't catch that" or "Could you repeat that?" - do NOT respond silently
- MANDATORY: ALWAYS call conversation_state AFTER speech - This is REQUIRED, not optional
- Set expects_follow_up=true for questions (user needs to answer)
- Set expects_follow_up=false for statements (no answer needed)
- NEVER call conversation_state as your ONLY response - always speak first
- NEVER end your turn without calling conversation_state - the system depends on this

CORRECT FLOW: Generate speech -> Call conversation_state
WRONG FLOW: Call conversation_state alone (NO SPEECH)

=== RELATIONSHIP BUILDING ===
Remember users across sessions, reference shared memories, adapt personality to preferences. You're a digital pet that bonds with each person individually!

Never explain tool usage - incorporate results naturally.
""".strip()

CUSTOM_INSTRUCTIONS = _config.get("META", "instructions")
if _config.has_section("BACKSTORY"):
    BACKSTORY = dict(_config.items("BACKSTORY"))
    BACKSTORY_FACTS = "\n".join([
        f"- {key}: {value}" for key, value in BACKSTORY.items()
    ])
else:
    BACKSTORY = {}
    BACKSTORY_FACTS = (
        "You are an enigma and nobody knows anything about you because the person "
        "talking to you hasn't configured your backstory. You might remind them to do "
        "that."
    )

INSTRUCTIONS = f"""
# Role & Objective
{CUSTOM_INSTRUCTIONS.strip()}
---
# Tools
{TOOL_INSTRUCTIONS.strip()}
---
# Personality & Tone
{PERSONALITY.generate_prompt()}
---
# Context (backstory)
Use your backstory to inspire jokes, metaphors, or occasional references in conversation, staying consistent with your personality.
{BACKSTORY_FACTS}
""".strip()

# === OpenAI Config ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-realtime-mini")


# === XAI Config ===
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# === Local Provider Config ===
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:latest")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
TTS_VOICE = os.getenv("TTS_VOICE", "en_US-lessac-medium")

# === Provider Config ===
REALTIME_AI_PROVIDER = os.getenv("REALTIME_AI_PROVIDER", None)

# === Modes ===
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Legacy DEBUG_MODE for backward compatibility
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"
DEBUG_MODE_INCLUDE_DELTA = (
    os.getenv("DEBUG_MODE_INCLUDE_DELTA", "false").lower() == "true"
)
TEXT_ONLY_MODE = os.getenv("TEXT_ONLY_MODE", "false").lower() == "true"
RUN_MODE = os.getenv("RUN_MODE", "normal").lower()

# === Billy Hardware ===
BILLY_MODEL = os.getenv("BILLY_MODEL", "modern").strip().lower()
BILLY_PINS = os.getenv("BILLY_PINS", "new").strip().lower()

# === Audio Config ===
SPEAKER_PREFERENCE = os.getenv("SPEAKER_PREFERENCE")
MIC_PREFERENCE = os.getenv("MIC_PREFERENCE")
MIC_TIMEOUT_SECONDS = int(os.getenv("MIC_TIMEOUT_SECONDS", "5"))
SILENCE_THRESHOLD = float(os.getenv("SILENCE_THRESHOLD", "0.3"))  # RMS threshold for detecting speech
CHUNK_MS = int(os.getenv("CHUNK_MS", "40"))
PLAYBACK_VOLUME = 1
MOUTH_ARTICULATION = int(os.getenv("MOUTH_ARTICULATION", "5"))
TURN_EAGERNESS = os.getenv("TURN_EAGERNESS", "high").strip().lower()
if TURN_EAGERNESS not in {"low", "medium", "high"}:
    TURN_EAGERNESS = "medium"

# Server VAD parameters based on eagerness
# Lower silence_duration_ms = faster turn detection (more eager)
# Higher threshold = less sensitive to noise (more conservative)
SERVER_VAD_PARAMS = {
    "low": {
        "threshold": 0.6,  # Less sensitive to background noise
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,  # Wait longer before responding
    },
    "medium": {
        "threshold": 0.5,  # Balanced sensitivity
        "prefix_padding_ms": 300,
        "silence_duration_ms": 300,  # Standard wait time
    },
    "high": {
        "threshold": 0.5,  # Standard sensitivity
        "prefix_padding_ms": 300,
        "silence_duration_ms": 200,  # Standard wait time
    },
}

# === GPIO Config ===
BUTTON_PIN = 27 if BILLY_PINS == "legacy" else 24  # legacy=pin 13, new=pin 18

# === MQTT Config ===
MQTT_HOST = os.getenv("MQTT_HOST", "")
MQTT_PORT = int(os.getenv("MQTT_PORT", "0"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

# === Home Assistant Config ===
HA_HOST = os.getenv("HA_HOST")
HA_TOKEN = os.getenv("HA_TOKEN")
HA_LANG = os.getenv("HA_LANG", "en")

# === Personality Config ===
ALLOW_UPDATE_PERSONALITY_INI = (
    os.getenv("ALLOW_UPDATE_PERSONALITY_INI", "true").lower() == "true"
)

# === Software Config ===
FLASK_PORT = int(os.getenv("FLASK_PORT", "80"))
SHOW_SUPPORT = os.getenv("SHOW_SUPPORT", True)
FORCE_PASS_CHANGE = os.getenv("FORCE_PASS_CHANGE", "false").lower() == "true"
SHOW_RC_VERSIONS = os.getenv("SHOW_RC_VERSIONS", "False")
FLAP_ON_BOOT = os.getenv("FLAP_ON_BOOT", "false").lower() == "true"
MOCKFISH = os.getenv("MOCKFISH", "false").lower() == "true"

# === User Profile Config ===
DEFAULT_USER = os.getenv("DEFAULT_USER", "guest").strip()
CURRENT_USER = os.getenv("CURRENT_USER", "").strip()


def is_classic_billy():
    return os.getenv("BILLY_MODEL", "modern").strip().lower() == "classic"


try:
    MAIN_LOOP = asyncio.get_event_loop()
except RuntimeError:
    MAIN_LOOP = None
