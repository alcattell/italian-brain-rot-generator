# Italian Brainrot Generator

This project generates surreal **Italian Brainrot characters** from any user input.  
It uses an **LLM with a production-grade system prompt** and falls back to a local rule-based generator if the API fails.

---

## Features
- Reads the system prompt from a `.txt` file (e.g. `brainrot_system_prompt.txt`).
- Calls OpenAI or Anthropic models to generate output in strict JSON format.
- Safety filter for politics, religion, brands, and other sensitive terms.
- Fallback to a deterministic local generator so you always get a result.
- CLI-friendly: pass in any text and get a Brainrot character back.

---

## Files
- `brainrot_llm.py` → Main script (CLI + generator logic).
- `brainrot_system_prompt.txt` → System prompt with full instructions for the LLM.

---

## Requirements
- Python 3.9+
- Install dependencies:

```bash
pip install openai anthropic
```

- Set an API key:

```bash
# For OpenAI
export OPENAI_API_KEY="sk-your-openai-key"

# For Anthropic
export ANTHROPIC_API_KEY="sk-your-anthropic-key"
```

---

## Usage

### Basic
```bash
python brainrot_llm.py "toothbrush" --provider openai --model gpt-5
```

### Anthropic
```bash
python brainrot_llm.py "toothbrush" --provider anthropic --model claude-3.7-sonnet
```

### With a random seed
```bash
python brainrot_llm.py "skateboard" --provider openai --model gpt-5 --seed 42
```

---

## Example Output
```json
{
  "name": "Spazzolinino Delfinino",
  "chant": "Spi spi spazzolà, delfinì!",
  "visual": "A dolphin with bristle-fin flippers and a mint-foam blowhole glides on toothpaste waves.",
  "lore": "Cleans seas and smiles."
}
```

---

## Prompt File
The system prompt lives in `brainrot_system_prompt.txt`.  
This is a long set of rules and examples that ensure the LLM produces on-style Brainrot characters.  
You can edit this file to tweak the style or add new few-shot examples.

---

## Fallback Mode
If the API is unavailable or the model outputs invalid JSON, the script falls back to a local rule-based generator that combines animals, foods, objects, and suffixes to produce Brainrot-like results.

---

## License
MIT – use, modify, and share freely.
