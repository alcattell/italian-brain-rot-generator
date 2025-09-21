#!/usr/bin/env python3
"""
brainrot_llm.py
Production generator for Italian Brainrot characters.

Features
- Reads system prompt from a .txt file (default: brainrot_system_prompt.txt)
- Calls an LLM (OpenAI or Anthropic; choose via --provider)
- Enforces JSON output schema and auto-repairs common formatting errors
- Safety filter for brands, politics, religion, hate terms
- Deterministic option via --seed (passed in user message and used in fallback)
- Fallback to local rule-based generator if API fails or returns invalid output
- CLI usage:
    python brainrot_llm.py "toothbrush" --provider openai --model gpt-5 --seed 7
"""

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Dict, Any, Optional

# ---------------------------
# Config
# ---------------------------

DEFAULT_PROMPT_PATH = "brainrot_system_prompt.txt"
DEFAULT_PROVIDER = "openai"          # options: openai | anthropic
DEFAULT_OPENAI_MODEL = "gpt-5"       # adjust to your deployment
DEFAULT_ANTHROPIC_MODEL = "claude-3.7-sonnet"  # adjust if needed
DEFAULT_TEMPERATURE = 1.1
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.5

# Minimal safety lists; extend for your needs
SENSITIVE = {
    "politics": {"putin","trump","biden","xi","modi","netanyahu","gaza","hamas","israel","palestine","isis","nazi","hitler"},
    "religion": {"jesus","christ","muhammad","allah","yahweh","buddha"},
    "brands": {"google","apple","nike","adidas","coca","mcdonalds","tiktok","instagram","meta","twitter","x"},
    "adult": {"sex","porn","nsfw"},
    "hate": {"slur1","slur2"}  # placeholder; populate as required
}
ALL_SENSITIVE = set().union(*SENSITIVE.values())

# ---------------------------
# Local fallback generator
# ---------------------------

ANIMALS = ["gatto","cane","delfino","polpo","coccodrillo","elefante","gufo","scimmia","orso","rana","tartaruga","volpe","pappagallo","riccio","ippopotamo"]
FOODS   = ["gelato","pizza","tiramisù","mortadella","pesto","maccheroni","ravioli","cappuccino","espresso","pomodoro","pistacchio","polpetta","panettone","cannelloni","risotto"]
OBJECTS = ["lampadina","scarpa","mandolino","ombrello","valigia","occhiali","trombetta","telefono","televisore","frullatore","spazzolino","computer","orologio","caramella","pennello"]
SUFFIX  = ["ini","oni","etta","etto","uccio","uzzo","ino","one","ello","ella","etti","uzzi"]
NOISES  = ["tra","la","brr","pata","pim","tini","tino","liri","là","pipo","rino","zuzzo","dada","miao","bau","plip","plop","dili","dilo"]

def _safe_tokens(text: str):
    return re.findall(r"[a-zA-Z]+", text.lower())

def _choose_coherent(user_tokens):
    vocab = set(user_tokens)
    if {"brush","tooth","dental"} & vocab: obj = "spazzolino"
    elif {"lamp","light","bulb"} & vocab: obj = "lampadina"
    elif {"phone","mobile","cell"} & vocab: obj = "telefono"
    elif {"umbrella","rain"} & vocab: obj = "ombrello"
    elif {"coffee","espresso","cappuccino"} & vocab: obj = "cappuccino"
    else:
        obj = random.choice(FOODS + OBJECTS)
    animal = random.choice(ANIMALS)
    thing = obj
    twist = random.choice([
        "three extra legs",
        "cactus skin",
        "neon whiskers",
        "espresso-steam breath",
        "pasta-stilt legs",
        "velcro feathers",
        "marble-polished shell",
        "bubblewrap scales",
        "clockwork tail",
        "accordion ribs",
    ])
    return animal, thing, twist

def _italianise(word: str):
    w = re.sub(r"[^a-z]", "", word.lower())
    w = w.replace("ph","f").replace("sh","s").replace("th","t")
    if not w:
        w = "liri"
    if not any(w.endswith(s) for s in SUFFIX):
        w += random.choice(SUFFIX)
    return w

def _stem(w: str):
    return re.sub(r"(ini|oni|etta|etto|uccio|uzzo|ino|one|ello|ella|etti|uzzi)$", "", w)

def _core_syllables(n=2):
    CV = ["la","li","lo","ra","ri","ro","na","ni","no","ta","ti","to","pa","pi","po","ca","co","cu","za","zi"]
    return "".join(random.choice(CV) for _ in range(n))

def _make_name(animal, thing):
    baseA = _italianise(animal)
    baseB = _italianise(thing)
    mode = random.choice(["rhyme","alliterate","redup","compound"])
    if mode == "rhyme":
        suf = random.choice(SUFFIX)
        A = _stem(baseA) + suf
        B = _stem(baseB) + suf
        return _cap(f"{A} {B}")
    if mode == "alliterate":
        ini = random.choice("bcdfglmprstvz")
        A = ini + _core_syllables(2)
        B = ini + _core_syllables(2)
        return _cap(f"{_italianise(A)} {_italianise(B)}")
    if mode == "redup":
        X = _cap(_core_syllables(3))
        Y = _cap(_core_syllables(3))
        return f"{X} {Y}"
    suf = random.choice(SUFFIX)
    return _cap(f"{_stem(baseB)} {_stem(baseA)}{suf}")

def _cap(s: str):
    return " ".join(p.capitalize() for p in s.split())

def _make_chant(name: str):
    parts = name.lower().split()
    pick = random.sample(NOISES, k=2) + [parts[0][:4]]
    chant = " ".join([*pick, random.choice(NOISES), random.choice(NOISES)])
    toks = chant.split()
    return " ".join(toks[:random.randint(3,8)])

def _make_visual(animal, thing, twist):
    textures = ["glossy","velvety","crispy","foamy","sparkling","marble","silky","rubbery","sticky","peppered"]
    motions   = ["glides","bounces","whirs","tiptoes","pirouettes","rattles","shuffles","zips","waddles","hums"]
    colours   = ["emerald","saffron","cobalt","scarlet","mint","ivory","charcoal","amber","pearl","neon-violet"]
    t = random.choice(textures)
    m = random.choice(motions)
    c = random.choice(colours)
    s1 = f"A {animal} with {t} {thing} parts and {twist} {m} across {c} confetti."
    s2 = "Its breath sounds like tiny tambourines."
    return s1 if random.random() < 0.6 else f"{s1} {s2}"

def local_fallback(user_input: str, seed: Optional[int]=None) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
    tokens = _safe_tokens(user_input)
    animal, thing, twist = _choose_coherent(tokens)
    name = _make_name(animal, thing)
    return {
        "name": name,
        "chant": _make_chant(name),
        "visual": _make_visual(animal, thing, twist),
        "lore": random.choice([
            "Collects lost spoons.",
            "Sings in pasta alleys.",
            "Guards midnight snacks.",
            "Polishes moonbeams.",
            "Chases bubble echoes."
        ])
    }

# ---------------------------
# Prompt I/O
# ---------------------------

def load_system_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        sys.stderr.write(f"[error] Prompt file not found: {path}\n")
        sys.exit(1)

# ---------------------------
# Safety + Validation
# ---------------------------

def hits_sensitive_language(text: str) -> bool:
    toks = set(_safe_tokens(text))
    return bool(toks & ALL_SENSITIVE)

def validate_and_repair_json(raw: str) -> Optional[Dict[str, Any]]:
    # Extract the largest JSON block
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        candidate = match.group(0)
    else:
        candidate = raw

    # Common fixes
    candidate = candidate.strip()
    candidate = candidate.replace("\u201c", "\"").replace("\u201d", "\"").replace("\u2019", "'")
    # Try parse
    try:
        data = json.loads(candidate)
    except Exception:
        return None

    # Schema
    if not isinstance(data, dict):
        return None
    for k in ("name","chant","visual"):
        if k not in data or not isinstance(data[k], str) or not data[k].strip():
            return None
    # Optional lore normalisation
    if "lore" in data and not isinstance(data["lore"], str):
        data["lore"] = str(data["lore"])

    # Safety: if any sensitive tokens appear, reject so we can retry
    if hits_sensitive_language(json.dumps(data)):
        return None

    # Hard limits
    if len(data["chant"].split()) > 12:
        data["chant"] = " ".join(data["chant"].split()[:8])
    if len(data["visual"]) > 400:
        data["visual"] = data["visual"][:400].rstrip() + "."

    return data

# ---------------------------
# LLM Clients
# ---------------------------

def call_openai(system_prompt: str, user_input: str, model: str, temperature: float) -> str:
    # Requires: pip install openai ; env var OPENAI_API_KEY
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(f"OpenAI client not available: {e}")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    # Use Chat Completions or Responses, depending on your SDK.
    # Here we use chat.completions for broad compatibility.
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    return resp.choices[0].message.content

def call_anthropic(system_prompt: str, user_input: str, model: str, temperature: float) -> str:
    # Requires: pip install anthropic ; env var ANTHROPIC_API_KEY
    try:
        import anthropic
    except Exception as e:
        raise RuntimeError(f"Anthropic client not available: {e}")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=600,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role":"user","content": user_input}]
    )
    # Concatenate content blocks to text
    parts = []
    for b in msg.content:
        if getattr(b, "type", None) == "text":
            parts.append(b.text)
        else:
            # anthropic SDK v1 returns dict-like items too
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text",""))
    return "\n".join(parts).strip()

# ---------------------------
# Orchestrator
# ---------------------------

def generate_brainrot(
    user_input: str,
    system_prompt_path: str = DEFAULT_PROMPT_PATH,
    provider: str = DEFAULT_PROVIDER,
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = None,
    retries: int = MAX_RETRIES
) -> Dict[str, Any]:

    system_prompt = load_system_prompt(system_prompt_path)

    # Propagate seed hint to the model for soft determinism
    user_message = user_input if seed is None else f"{user_input}\n\n[seed:{seed}]"

    if provider == "openai":
        model = model or DEFAULT_OPENAI_MODEL
        caller = lambda: call_openai(system_prompt, user_message, model, temperature)
    elif provider == "anthropic":
        model = model or DEFAULT_ANTHROPIC_MODEL
        caller = lambda: call_anthropic(system_prompt, user_message, model, temperature)
    else:
        raise ValueError("Unsupported provider. Use 'openai' or 'anthropic'.")

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            raw = caller()
            parsed = validate_and_repair_json(raw)
            if parsed:
                return parsed
            last_error = f"invalid_json_attempt_{attempt}"
        except Exception as e:
            last_error = str(e)
        time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    # Fallback
    fb = local_fallback(user_input, seed=seed)
    fb["_meta"] = {"source": "fallback", "last_error": last_error}
    return fb

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Italian Brainrot LLM generator")
    ap.add_argument("prompt", help="User input")
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPT_PATH, help="Path to system prompt .txt")
    ap.add_argument("--provider", choices=["openai","anthropic"], default=DEFAULT_PROVIDER)
    ap.add_argument("--model", default=None, help="Model name (overrides default)")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--retries", type=int, default=MAX_RETRIES)
    args = ap.parse_args()

    result = generate_brainrot(
        user_input=args.prompt,
        system_prompt_path=args.prompt_file,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        retries=args.retries
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
