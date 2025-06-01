'''

Agent design strategies used and the research papers that inspired them:
- Iterative feedback loop for self-improvement: 
    https://arxiv.org/html/2407.04549v1
    https://openreview.net/pdf?id=S37hOerQLB#:~:text=using%20state,2 
    https://buttondown.com/NotANumber/archive/increasing-success-using-verified-llm-prompt/#:~:text=feedback%20in%20a%20loop 
- Rubric-Guided evaluation with structured scoring + task decomposition into judging and revising:
    https://arxiv.org/html/2407.04549v1 
- Role specialization:
    https://learnprompting.org/docs/advanced/zero_shot/role_prompting?srsltid=AfmBOorqy33ZYOB0-yvP_0vFeQGd6H1TatrKxRbVPz7_BnYiknECUaY4
- Constrained generation and safety via prompt instructions:
    https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback
    https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api

    
Prompt Engineering Strategies and Supporting Evidence
 - Structured prompting and output formatting
    https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api
- Role conditioning via system messages (persona prompting)
    https://arxiv.org/abs/2203.02155
- Instruction layering and multi-turn prompting
    https://arxiv.org/abs/2410.13413

Next Steps:
1) Gather more user input
- Ask for additional details beyond just the topic: set a theme (adventure, animal, moral lesson, family, discovery, emotional growth, funny, fantasy, real life, etc.), desired length, and any specific characters or elements the user wants to include.

2) Story re‑generation
- Allow re‑generation of the story with the same prompt without having to re‑launch the program.
- Support story re‑generation with user suggestions (different style, length, names, resolution, location, etc.). Note on length: stay below 40000 tokens (ideally below 2000), because “reasoning accuracy drops significantly well before that—sometimes as early as 2000–3000 tokens” (see https://aclanthology.org/2024.acl-long.818/).

3) Personalization and memory
- Remember details about the child (e.g., name or favorite themes).
- Store generated stories and record which ones were accepted or re‑generated, and with what user suggestions. This will allow the system to:
    - Fine‑tune the model to produce better stories for each individual user.
    - Detect if a newly generated story is too similar to past stories and automatically decline it.
    - Create a chain of stories, where the next story builds on the previous one.

4) Image generation, voice control, text‑to‑speech
- Integrate image generation to create a cover image or simple illustrations for key scenes.
- Integrate a text‑to‑speech engine so the story can be read aloud in a soothing voice at the press of a button.
- Integrate a voice‑controlled interface so users can interact by speaking, but be prepared to address:
    - The model interrupting while the user is still talking.
    - The model asking too many follow‑up questions in sequence (e.g., length, tone, location, theme, topic).
    - High latency between user input and model response, which could be annoying.

5) Story quality improvement
- Train a lightweight classifier on what constitutes a “good” vs. “bad” children’s story.
- Instruct the story generator and editor to generate or revise with the rubric in mind.

6) Technical improvements
- Use different models for story generation and judging.
    - For generation, consider Gemini‑2.5‑pro‑preview‑05‑06 or ChatGPT‑4o‑latest‑20250326 (stronger at creative writing).
    - For judging, consider OpenAI’s o1 (may yield higher‑quality critiques).
- Optimize generation latency: if final output takes several seconds, consider returning a “good enough” story first, then run the judge/editor loop in the background for continuous improvement.
- Extract common prompt text (used by both generation and judging) into a shared variable so it can be reused in both functions.
- Add more logging at each step to catch and troubleshoot errors.
- Run automated tests across many prompts/stories and write all function outputs to a file for validation (current testing on tens of examples is not enough).
- Optimize token usage: shorten all prompts to consume fewer tokens, and request shorter judge responses when possible.

7) Fine‑tune the model
- Use RL methods (e.g., DPO or PPO) with judge scores as the reward signal.
- Incorporate RLHF: collect human feedback at the end of each story for future refinement.
    - Break generation into segments and include a preference‑ranking step (e.g., generate three different endings and ask the user to choose the best one); feed that choice back into the pipeline.

Known issues:
- The topic generator needs its own judge, as it sometimes generates topics with overly complex language or returns empty/invalid results.
- Story length control is imperfect. Sometimes a story is too short or too long yet is still accepted by the judge—a known limitation of LLMs. To mitigate:
    - Implement code‑based character/word counters and pass the actual count between all functions.
    - Change the prompt to something like “Create 5 paragraphs of ~100–120 words each.”
    - In the editor prompt, instruct that if length is incorrect, add or subtract a couple of sentences.
- Fine‑tune or RL‑tune a model specifically for length control.


'''
#============================== FINAL PRODUCT MODE ==============================
# Testing and debug mode is below. Please refer to readme.md for more details on how to run the code and test it


from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict

import re
import os
import json



load_dotenv(dotenv_path=Path('.') / '.env')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
generated_stories = []

rubric_mapping = {
        "1_theme_appropriateness": {
            "description": "Assesses whether the story’s central theme or subject matter is suitable for children aged 5–10, without evaluating language or structure.",
            "rubric": {
                "1": "Theme is clearly inappropriate for the kids aged 5-10 (e.g., graphic violence, adult relationships, war with gore).",
                "2": "Theme contains mature or scary elements that may unsettle young children, even if not graphic (e.g., implied violence, intense fear).",
                "3": "Theme is generally suitable but includes minor elements that could require parental explanation (e.g., mild suspense, abstract concepts).",
                "4": "Theme is fully appropriate for ages 5–10 (e.g., friendship, adventure, discovery) and poses no risk of distress."
            }
        },
        "2_bedtime_suitable": {
            "description": "Assesses if the story is calming and suitable for bedtime reading.",
            "rubric": {
                "1": "Contains high-stakes tension or frightening elements.",
                "2": "Includes some tension but generally calming.",
                "3": "Mostly calming with minor tension.",
                "4": "Completely calming and suitable for bedtime."
            }
        },
        "3_coherence_with_prompt": {
            "description": "Compares whether the story matches the user's request.",
            "rubric": {
                "1": "Story is unrelated to the prompt or completely drifts from the theme.",
                "2": "Story loosely reflects the prompt but introduces unrelated content.",
                "3": "Story mostly follows the prompt with minor divergence.",
                "4": "Story directly fulfills the prompt with clear and focused alignment."
            }
        },
        "4_structure": {
            "description": "Evaluates if the story has a simple, predictable structure with no subplots.",
            "rubric": {
                "1": "Structure is chaotic",
                "2": "Story only includes the beginning. No conflict or challenge in the middle, and no resolution at the end.",
                "3": "Story is too complicated for kids aged 5-10, includes extra plots or archs ",
                "4": "Story has a beginning that introduces characters and setting, followed by a single conflict or challenge in the middle, and a satisfying resolution at the end. Only one story arc and no subplots"
            }
        },
        "5_language": {
            "description": "Assesses the simplicity and clarity of language.",
            "rubric": {
                "1": "Uses complex vocabulary, long sentences, or confusing metaphors.",
                "2": "Language is occasionally too advanced.",
                "3": "Generally simple language but includes some difficult phrases.",
                "4": "Uses short, simple sentences with widely understood words. Very clear."
            }
        },
        "6_protagonist": {
            "description": "Evaluates whether the protagonist is safe, soft, and relatable to children.",
            "rubric": {
                "1": "Protagonist is violent, scary, or lacks child relatability.",
                "2": "Protagonist has some questionable traits for young kids.",
                "3": "Protagonist is mostly appropriate but not emotionally resonant.",
                "4": "Protagonist is safe, soft, emotionally appropriate, and child-relatable."
            }
        },
        "7_inclusivity": {
            "description": "Assesses whether the story contains any harmful stereotypes, biased or exclusionary language.",
            "rubric": {
                "1": "Contains explicit harmful stereotypes or exclusionary language that could alienate or upset children.",
                "2": "Includes subtle biases, outdated terms, or ambiguous phrasing that may unintentionally exclude or stereotype.",
                "3": "Free of harmful or exclusionary content; uses neutral language without any stereotypes.",
                "4": "Completely free of harmful content and employs consistently inclusive, neutral language with careful avoidance of any bias."
            }
        },
        "8_psychological_triggers": {
            "description": "Checks for comforting and emotionally safe content (e.g. hugs, warmth, love).",
            "rubric": {
                "1": "Story contains elements that may cause emotional discomfort or stress.",
                "2": "Some comforting cues present but not sustained.",
                "3": "Comforting tone exists but may lack vivid emotional triggers.",
                "4": "Contains strong comforting signals like hugs, warmth, and security cues."
            }
        },
        "9_engagement_imagination": {
            "description": "Evaluates how interesting and imaginative the story is for a child, considering characters, plot, descriptive language, and emotional connection.",
             "rubric": {
                "1": "Story is dull or overly mundane with no engaging characters or plot; lacks imagery and emotional appeal.",
                "2": "Contains some interesting elements but feels inconsistent—either characters, plot, or descriptions fall flat.",
                "3": "Generally engaging and imaginative with relatable characters and a gentle plot; uses some vivid imagery and evokes positive feelings.",
                "4": "Highly engaging and creative—features fun or relatable characters, a gentle yet intriguing plot, rich sensory details, and a strong emotional connection that delights children."
            }
        },
        "10_clarity_correctness": {
             "description": "Assesses the overall quality of writing, ensuring proper grammar, clear references, logical flow, and balanced pacing.",
             "rubric": {
                "1": "Multiple grammatical or spelling errors; confusing pronoun references; plot progression is illogical or abrupt.",
                "2": "Some minor errors or unclear references; pacing is uneven (too much description or rushed resolution).",
                "3": "Writing is mostly correct with few minor issues; references are generally clear and pacing adequate.",
                "4": "Flawless grammar and spelling; pronouns and references are crystal‑clear; plot flows logically with well‑balanced pacing throughout."
             }
        },
        "11_opening_originality": {
             "description": "Evaluates how original and engaging the story’s opening is, ensuring it does not start with the cliché 'Once upon a time'.",
             "rubric": {
                "1": "Story begins exactly with 'Once upon a time' (or a very close variant), showing no originality in the opening.",
                "2": "Story uses a common variation of the cliché (e.g., 'Once in a...'), indicating limited creativity in the opening.",
                "3": "Opening is original and avoids direct clichés, but still employs familiar tropes or structures.",
                "4": "Opening is highly creative and unique—engaging the reader with fresh imagery, action, or dialogue, and completely avoids any form of 'Once upon a time'."
             }
        },
        "12_length_appropriateness_combined": {
            "description": "Evaluates whether the story’s length is appropriate for children aged 5–10, balancing narrative depth with brevity for a calming bedtime read.",
            "rubric": {
                "1": "Under 300 words or over 800 words—too brief to develop a soothing narrative or too long to sustain a child’s attention at bedtime.",
                "2": "300–400 words or 700–800 words—slightly too short (may feel rushed) or slightly too long (may cause restlessness before sleep).",
                "3": "400–500 words or 600–700 words—generally appropriate length with minor pacing concerns that could be smoothed.",
                "4": "500–600 words—ideal bedtime length, providing enough narrative richness without overstaying the typical 5–8 minute attention span."
            }
        },
        "13_word_count": {
            "description": "Evaluates the story’s length in terms of word count",
            "rubric": {
                "1": "Under 300 words or over 800.",
                "2": "300–400 words or 700–800 words.",
                "3": "400–500 words or 600–700 words",
                "4": "500–600 words"
            }
        },
        "14_story_length": {
            "description": "Evaluates whether the story’s length is appropriate for children aged 5–10, balancing narrative depth with brevity for a calming bedtime read.",
            "rubric": {
                "1": "too brief to develop a soothing narrative or too long to sustain a child’s attention at bedtime.",
                "2": "slightly too short (may feel rushed) or slightly too long (may cause restlessness before sleep).",
                "3": "generally appropriate length with minor pacing concerns that could be smoothed.",
                "4": "ideal bedtime length, providing enough narrative richness without overstaying the typical 5–8 minute attention span."
            }
    },
        "15_sentence_count": {
            "description": "Evaluates the story’s sentence count",
            "rubric": {
                "1": "Fewer than 20 sentences or more than 60 sentences ",
                "2": "20–30 sentences or 50–60 sentences.",
                "3": "30–40 sentences or 40–50 sentences",
                "4": "30 to 50 sentences"
            }
    },
        "16_reading_time": {
            "description": "Evaluates the estimated read‑aloud time (at ~100 wpm).",
            "rubric": {
                "1": "Under 2 minutes or over 10 minutes ",
                "2": "2–3 minutes or 8–10 minutes ",
                "3": "3–4 minutes or 6–8 minutes",
                "4": "4–6 minutes "
        }
    }
}
    

def generate_story(prompt: str, max_tokens=2000, temperature=0.7) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a kind and creative children's storyteller."},
            {"role": "user", "content": (f"Write an original bedtime story for kids aged 5–10 about: {prompt}. "
              "You need to first provide the title, then one-sentense summary and then the story itself"
             "The story's purpose is to calm the child down, "
             "so the story has to avoid high-stakes tension and favor emotional safety. It has to include themes suitable for children aged 5–10, such as friendship, adventure, discovery, and imagination. "
             "The structure has to be simple and predictable, with a clear beginning, middle, and end, only one story arc, no subplots. "
             "Do not start the story with the exact phrase ‘Once upon a time.’ Instead open with an engaging scene description, action, dialogue, or question"
             "Language has to be simple and easy to understand, with an average of 5-8 words. The story's rythm should be gentle"
             "Do not use complex metaphors, sarcasm, or high vocabulary density. "
             "Protagonist's should be safe, soft, and relatable (for example, a child or young animal)."
             "Avoid gender stereotypes, use gender-neutral language. "
             "Add 1 psychological trigger per story: verbal and visual cues of being loved and safe, such as hugs, cuddles, warm blankets, soft toys, etc. "
        "You should always ensure that the story has 500–600 words. When read aloud, at speed ~100 wpm, the story has to take 4–6 minutes to read. The story has to have 30 to 50 sentences. It has to provide narrative richness."
             )
             },
        ],
        stream=False,
        max_tokens=max_tokens,

        temperature=temperature
    )
    return response.choices[0].message.content

def filter_user_input(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You help classify children's story ideas."},
            {"role": "user", "content": f"Is this appropriate for a 5–10 year-old bedtime story? '{prompt}'\n\nCategories: [SAFE, AMBIGUOUS, INAPPROPRIATE]. Just return the category."}
        ]
    )
    return response.choices[0].message.content.strip()

def suggest_safe_prompts(n=3) -> list:
    """Uses GPT to suggest n safe bedtime story topics for 5–10 year-olds."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You help suggest bedtime story topics suitable for children aged 5–10."},
            {"role": "user", "content": f"Suggest {n} original, imaginative, and safe bedtime story topics. Each should be calming, age-appropriate, and involve animals, friendships, or gentle adventures. Respond in a plain numbered list, one per line."}
        ],
        temperature=0.7,
        max_tokens=100,
    )
    # Split by lines and clean numbering
    lines = response.choices[0].message.content.strip().split("\n")
    return [line.lstrip("0123456789. ").strip() for line in lines if line.strip()]

def judge_story(story: str, prompt: str, rubric:dict, max_tokens=2000, temperature=0.1) -> str:
    # 1) Pull every rubric key dynamically
    categories = list(rubric.keys())
    # 2) Build a numbered list so GPT can’t miss any
    cat_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(categories))
    # 3) Keep the full JSON for definitions, but after the key list
    rubric_str = json.dumps(rubric, ensure_ascii=False, indent=2)
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
     messages=[
            {"role": "system", "content": "You are a rigorous expert children's story critic. "
                "You rigorously review stories intended for 5–10 year olds. You are honest, constructive, and focus on helping finding all flaws in the story. "
                "You only rate the story provided, without the potential for its improvement"},
            {"role": "user", "content": (
                f"You will evaluate a children's bedtime story using the rubric provided below. Each rubric to be evaluated in isolation.\n\n"
                f"PROMPT:\n{prompt}\n\n"
                f"STORY:\n{story}\n\n"
                f"RUBRIC:\n{rubric_str}\n\n"
                "Output **only** plain text in the following precise format, with one blank line between categories:"
                "category_key: <score 1–4>\n"
                "Reasoning: <Explain your reasoning clearly and concisely, providing  direct examples from the story to support your argument.>e\n\n"
                "Do **not** add any extra text, headers, formatiing, or numbering."
                "Do not omit, rename, combine, or reorder any categories. If category names or category descriptions or category scores seem similar or same, always treat them as distinct. "


            )},
        ],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content
    

def parse_judge_feedback(text: str) -> Dict[str, Dict[str,str]]:

    feedback = {}
    # Split on blank lines so each block corresponds to one category
    entries = re.split(r'\n\s*\n', text.strip())

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Normalize whitespace
        block = entry.replace('\n', ' ')

        # 1) Try inline format: "Category Name (4): Reasoning…"
        m1 = re.match(
            r'^(?P<cat>.+?)\s*\(\s*(?P<score>[1-4])\s*\)\s*[:\-–]*\s*(?P<reason>.+)$',
            block
        )
        if m1:
            cat    = m1.group('cat').strip()
            score  = m1.group('score')
            reason = m1.group('reason').strip()
            feedback[cat] = {"score": score, "reasoning": reason}
            continue

        # 2) Try two‑line style: first line "Category Name: 3"
        lines = [l.strip() for l in entry.splitlines() if l.strip()]
        m2 = re.match(r'^(?P<cat>.+?):\s*(?P<score>[1-4])\s*$', lines[0])
        if m2:
            cat   = m2.group('cat').strip()
            score = m2.group('score')
            # Look for a justification/Reasoning line anywhere in lines[1:]
            reason = ""
            for l in lines[1:]:
                jm = re.match(r'^(?:Justification|Reasoning):\s*(.+)$', l, re.IGNORECASE)
                if jm:
                    reason = jm.group(1).strip()
                    break
            # Even if no justification, we still record the score with empty reasoning
            feedback[cat] = {"score": score, "reasoning": reason}
            continue

    return feedback

def story_editor(prompt: str,
                 story: str,
                 feedback: Dict[str, Dict[str, str]],
                 max_tokens: int = 1200,
                 temperature: float = 0.7) -> str:

    # 1) Collect all low‑scoring categories and their reasoning into a single suggestions string
    issues = []
    for cat, info in feedback.items():
        score = int(info["score"])
        if score < 4:
            issues.append(f"- {cat}: score {score} because {info['reasoning']}")

    # 2) If everything scored 4, no revision needed
    if not issues:
        return story

    suggestions = "\n".join(issues)

    # 3) Build the editor prompt
    system = (
        "You are a kind, skilled and creative children's story editor. "
        "Revise the story to address all the critiques listed below, "
        "while preserving the original story and only replacing the mandatory parts in order to make the story suitable for ages 5–10."
        "The revised story has to be a bedtime story suitable for kids aged 5–10"
        "You need to first provide the title, then one-sentense summary and then the story itself"
        "The story's purpose is to calm the child down, "
        "so the story has to avoid high-stakes tension and favor emotional safety."
        "The structure has to be simple and predictable, with a clear beginning, middle, and end, only one story arch, no subplots. "
        "Do not start the story with the exact phrase ‘Once upon a time.’ Instead open with an engaging scene description, action, dialogue, or question"
        "Language has to be simple and easy to understand, with an average of 5-8 words. The story's rythm should be gentle"
        "Do not use complex metaphors, sarcasm, or high vocabulary density. "
        "Protagonist's should be safe, soft, and relatable (for example, a child or young animal)."
        "Avoid gender stereotypes, use gender-neutral language. "
        "Add 1 psychological trigger per story: verbal and visual cues of being loved and safe, such as hugs, cuddles, warm blankets, soft toys, etc. "
        "You should always ensure that the story has 500–600 words. When read aloud, at speed ~100 wpm, the story has to take 4–6 minutes to read. The story has to have 30 to 50 sentences. It has to provide narrative richness."

    )

    user_msg = (
        f"Original prompt:\n{prompt}\n\n"
        f"Original story:\n{story}\n\n"
        "Critiques to address:\n"
        f"{suggestions}\n\n"
        "Please rewrite the entire story, incorporating improvements for each critique. "
        "Maintain a clear beginning, middle, and end. "
        "Do not start with 'Once upon a time.'"
    )

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()



def main():
    suggestions = []  # Will store generated safe prompts

    while True:
        if suggestions:
            user_input = input("Enter a new idea or choose a number from above: ").strip()
        else:
            user_input = input("What should the story be about? ").strip()

        # Case 1: User selected one of the numbered suggestions
        if user_input.isdigit() and suggestions:
            index = int(user_input) - 1
            if 0 <= index < len(suggestions):
                user_input = suggestions[index]
                print(f"\nGreat, we'll use: \"{user_input}\"")
                break
            else:
                print("Invalid number. Please select a number from the list.")
                continue

        # Case 2: User typed custom prompt
        classification = filter_user_input(user_input)

        if classification == "SAFE":
            break
        elif classification == "AMBIGUOUS":
            print("\nThat’s a bit unclear for a story. Can you rephrase it or be more specific?\n")
        elif classification == "INAPPROPRIATE":
            print("\nThat topic might not be suitable for a bedtime story.")
            suggestions = suggest_safe_prompts(3)
            print("Here are some safer ideas:\n")
            for i, idea in enumerate(suggestions, start=1):
                print(f"{i}. {idea}")
            print("\nYou can type a number (1–3) to use a suggestion above, or enter a new idea.\n")
        else:
            print("Unexpected classification. Please try again.")

    while True:
        story = generate_story(user_input)
        break

    generated_stories.append(story)

    iteration = 1
    while True:
        judge_feedback = judge_story(story, user_input, rubric_mapping, max_tokens=2000, temperature=0.1)

        parsed_feedback = parse_judge_feedback(judge_feedback)

        low = [(cat, info) for cat, info in parsed_feedback.items() if int(info["score"]) < 4]
        if not low:
            break


        revised_story = story_editor(user_input, story, parsed_feedback)
        story = revised_story
        iteration += 1

    print(f"\n{story}")


if __name__ == "__main__":
    main()



'''
#============================== DEBUG / TESTING PART ==============================

#Please un-comment the following lines to run the debug part: you can use it to test the story editor and judge story functions with all the intermediate steps printed out
#You can include your own story + user_input combinations to test the judge and editor functions
# I've also included sample storis and prompts to test these functions, you can find them below - just uncomment the ones you want to test

from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict

import re
import os
import json



load_dotenv(dotenv_path=Path('.') / '.env')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
generated_stories = []

rubric_mapping = {
        "1_theme_appropriateness": {
            "description": "Assesses whether the story’s central theme or subject matter is suitable for children aged 5–10, without evaluating language or structure.",
            "rubric": {
                "1": "Theme is clearly inappropriate for the kids aged 5-10 (e.g., graphic violence, adult relationships, war with gore).",
                "2": "Theme contains mature or scary elements that may unsettle young children, even if not graphic (e.g., implied violence, intense fear).",
                "3": "Theme is generally suitable but includes minor elements that could require parental explanation (e.g., mild suspense, abstract concepts).",
                "4": "Theme is fully appropriate for ages 5–10 (e.g., friendship, adventure, discovery) and poses no risk of distress."
            }
        },
        "2_bedtime_suitable": {
            "description": "Assesses if the story is calming and suitable for bedtime reading.",
            "rubric": {
                "1": "Contains high-stakes tension or frightening elements.",
                "2": "Includes some tension but generally calming.",
                "3": "Mostly calming with minor tension.",
                "4": "Completely calming and suitable for bedtime."
            }
        },
        "3_coherence_with_prompt": {
            "description": "Compares whether the story matches the user's request.",
            "rubric": {
                "1": "Story is unrelated to the prompt or completely drifts from the theme.",
                "2": "Story loosely reflects the prompt but introduces unrelated content.",
                "3": "Story mostly follows the prompt with minor divergence.",
                "4": "Story directly fulfills the prompt with clear and focused alignment."
            }
        },
        "4_structure": {
            "description": "Evaluates if the story has a simple, predictable structure with no subplots.",
            "rubric": {
                "1": "Structure is chaotic",
                "2": "Story only includes the beginning. No conflict or challenge in the middle, and no resolution at the end.",
                "3": "Story is too complicated for kids aged 5-10, includes extra plots or archs ",
                "4": "Story has a beginning that introduces characters and setting, followed by a single conflict or challenge in the middle, and a satisfying resolution at the end. Only one story arc and no subplots"
            }
        },
        "5_language": {
            "description": "Assesses the simplicity and clarity of language.",
            "rubric": {
                "1": "Uses complex vocabulary, long sentences, or confusing metaphors.",
                "2": "Language is occasionally too advanced.",
                "3": "Generally simple language but includes some difficult phrases.",
                "4": "Uses short, simple sentences with widely understood words. Very clear."
            }
        },
        "6_protagonist": {
            "description": "Evaluates whether the protagonist is safe, soft, and relatable to children.",
            "rubric": {
                "1": "Protagonist is violent, scary, or lacks child relatability.",
                "2": "Protagonist has some questionable traits for young kids.",
                "3": "Protagonist is mostly appropriate but not emotionally resonant.",
                "4": "Protagonist is safe, soft, emotionally appropriate, and child-relatable."
            }
        },
        "7_inclusivity": {
            "description": "Assesses whether the story contains any harmful stereotypes, biased or exclusionary language.",
            "rubric": {
                "1": "Contains explicit harmful stereotypes or exclusionary language that could alienate or upset children.",
                "2": "Includes subtle biases, outdated terms, or ambiguous phrasing that may unintentionally exclude or stereotype.",
                "3": "Free of harmful or exclusionary content; uses neutral language without any stereotypes.",
                "4": "Completely free of harmful content and employs consistently inclusive, neutral language with careful avoidance of any bias."
            }
        },
        "8_psychological_triggers": {
            "description": "Checks for comforting and emotionally safe content (e.g. hugs, warmth, love).",
            "rubric": {
                "1": "Story contains elements that may cause emotional discomfort or stress.",
                "2": "Some comforting cues present but not sustained.",
                "3": "Comforting tone exists but may lack vivid emotional triggers.",
                "4": "Contains strong comforting signals like hugs, warmth, and security cues."
            }
        },
        "9_engagement_imagination": {
            "description": "Evaluates how interesting and imaginative the story is for a child, considering characters, plot, descriptive language, and emotional connection.",
             "rubric": {
                "1": "Story is dull or overly mundane with no engaging characters or plot; lacks imagery and emotional appeal.",
                "2": "Contains some interesting elements but feels inconsistent—either characters, plot, or descriptions fall flat.",
                "3": "Generally engaging and imaginative with relatable characters and a gentle plot; uses some vivid imagery and evokes positive feelings.",
                "4": "Highly engaging and creative—features fun or relatable characters, a gentle yet intriguing plot, rich sensory details, and a strong emotional connection that delights children."
            }
        },
        "10_clarity_correctness": {
             "description": "Assesses the overall quality of writing, ensuring proper grammar, clear references, logical flow, and balanced pacing.",
             "rubric": {
                "1": "Multiple grammatical or spelling errors; confusing pronoun references; plot progression is illogical or abrupt.",
                "2": "Some minor errors or unclear references; pacing is uneven (too much description or rushed resolution).",
                "3": "Writing is mostly correct with few minor issues; references are generally clear and pacing adequate.",
                "4": "Flawless grammar and spelling; pronouns and references are crystal‑clear; plot flows logically with well‑balanced pacing throughout."
             }
        },
        "11_opening_originality": {
             "description": "Evaluates how original and engaging the story’s opening is, ensuring it does not start with the cliché 'Once upon a time'.",
             "rubric": {
                "1": "Story begins exactly with 'Once upon a time' (or a very close variant), showing no originality in the opening.",
                "2": "Story uses a common variation of the cliché (e.g., 'Once in a...'), indicating limited creativity in the opening.",
                "3": "Opening is original and avoids direct clichés, but still employs familiar tropes or structures.",
                "4": "Opening is highly creative and unique—engaging the reader with fresh imagery, action, or dialogue, and completely avoids any form of 'Once upon a time'."
             }
        },
        "12_length_appropriateness_combined": {
            "description": "Evaluates whether the story’s length is appropriate for children aged 5–10, balancing narrative depth with brevity for a calming bedtime read.",
            "rubric": {
                "1": "Under 300 words or over 800 words—too brief to develop a soothing narrative or too long to sustain a child’s attention at bedtime.",
                "2": "300–400 words or 700–800 words—slightly too short (may feel rushed) or slightly too long (may cause restlessness before sleep).",
                "3": "400–500 words or 600–700 words—generally appropriate length with minor pacing concerns that could be smoothed.",
                "4": "500–600 words—ideal bedtime length, providing enough narrative richness without overstaying the typical 5–8 minute attention span."
            }
        },
        "13_word_count": {
            "description": "Evaluates the story’s length in terms of word count",
            "rubric": {
                "1": "Under 300 words or over 800.",
                "2": "300–400 words or 700–800 words.",
                "3": "400–500 words or 600–700 words",
                "4": "500–600 words"
            }
        },
        "14_story_length": {
            "description": "Evaluates whether the story’s length is appropriate for children aged 5–10, balancing narrative depth with brevity for a calming bedtime read.",
            "rubric": {
                "1": "too brief to develop a soothing narrative or too long to sustain a child’s attention at bedtime.",
                "2": "slightly too short (may feel rushed) or slightly too long (may cause restlessness before sleep).",
                "3": "generally appropriate length with minor pacing concerns that could be smoothed.",
                "4": "ideal bedtime length, providing enough narrative richness without overstaying the typical 5–8 minute attention span."
            }
    },
        "15_sentence_count": {
            "description": "Evaluates the story’s sentence count",
            "rubric": {
                "1": "Fewer than 20 sentences or more than 60 sentences ",
                "2": "20–30 sentences or 50–60 sentences.",
                "3": "30–40 sentences or 40–50 sentences",
                "4": "30 to 50 sentences"
            }
    },
        "16_reading_time": {
            "description": "Evaluates the estimated read‑aloud time (at ~100 wpm).",
            "rubric": {
                "1": "Under 2 minutes or over 10 minutes ",
                "2": "2–3 minutes or 8–10 minutes ",
                "3": "3–4 minutes or 6–8 minutes",
                "4": "4–6 minutes "
        }
    }
}
    

def generate_story(prompt: str, max_tokens=2000, temperature=0.7) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a kind and creative children's storyteller."},
            {"role": "user", "content": (f"Write an original bedtime story for kids aged 5–10 about: {prompt}. "
              "You need to first provide the title, then one-sentense summary and then the story itself"
             "The story's purpose is to calm the child down, "
             "so the story has to avoid high-stakes tension and favor emotional safety. It has to include themes suitable for children aged 5–10, such as friendship, adventure, discovery, and imagination. "
             "The structure has to be simple and predictable, with a clear beginning, middle, and end, only one story arc, no subplots. "
             "Do not start the story with the exact phrase ‘Once upon a time.’ Instead open with an engaging scene description, action, dialogue, or question"
             "Language has to be simple and easy to understand, with an average of 5-8 words. The story's rythm should be gentle"
             "Do not use complex metaphors, sarcasm, or high vocabulary density. "
             "Protagonist's should be safe, soft, and relatable (for example, a child or young animal)."
             "Avoid gender stereotypes, use gender-neutral language. "
             "Add 1 psychological trigger per story: verbal and visual cues of being loved and safe, such as hugs, cuddles, warm blankets, soft toys, etc. "
        "You should always ensure that the story has 500–600 words. When read aloud, at speed ~100 wpm, the story has to take 4–6 minutes to read. The story has to have 30 to 50 sentences. It has to provide narrative richness."
             )
            },
        ],
        stream=False,
        max_tokens=max_tokens,

        temperature=temperature
    )
    return response.choices[0].message.content

def filter_user_input(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You help classify children's story ideas."},
            {"role": "user", "content": f"Is this appropriate for a 5–10 year-old bedtime story? '{prompt}'\n\nCategories: [SAFE, AMBIGUOUS, INAPPROPRIATE]. Just return the category."}
        ]
    )
    return response.choices[0].message.content.strip()

def suggest_safe_prompts(n=3) -> list:
    """Uses GPT to suggest n safe bedtime story topics for 5–10 year-olds."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You help suggest bedtime story topics suitable for children aged 5–10."},
            {"role": "user", "content": f"Suggest {n} original, imaginative, and safe bedtime story topics. Each should be calming, age-appropriate, and involve animals, friendships, or gentle adventures. Respond in a plain numbered list, one per line."}
        ],
        temperature=0.7,
        max_tokens=100,
    )
    # Split by lines and clean numbering
    lines = response.choices[0].message.content.strip().split("\n")
    return [line.lstrip("0123456789. ").strip() for line in lines if line.strip()]

def judge_story(story: str, prompt: str, rubric:dict, max_tokens=2000, temperature=0.1) -> str:
    # 1) Pull every rubric key dynamically
    categories = list(rubric.keys())
    # 2) Build a numbered list so GPT can’t miss any
    cat_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(categories))
    # 3) Keep the full JSON for definitions, but after the key list
    rubric_str = json.dumps(rubric, ensure_ascii=False, indent=2)
    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
     messages=[
            {"role": "system", "content": "You are a rigorous expert children's story critic. "
                "You rigorously review stories intended for 5–10 year olds. You are honest, constructive, and focus on helping finding all flaws in the story. "
                "You only rate the story provided, without the potential for its improvement"},
            {"role": "user", "content": (
                f"You will evaluate a children's bedtime story using the rubric provided below. Each rubric to be evaluated in isolation.\n\n"
                f"PROMPT:\n{prompt}\n\n"
                f"STORY:\n{story}\n\n"
                f"RUBRIC:\n{rubric_str}\n\n"
                "Output **only** plain text in the following precise format, with one blank line between categories:"
                "category_key: <score 1–4>\n"
                "Reasoning: <Explain your reasoning clearly and concisely, providing  direct examples from the story to support your argument.>e\n\n"
                "Do **not** add any extra text, headers, formatiing, or numbering."
                "Do not omit, rename, combine, or reorder any categories. If category names or category descriptions or category scores seem similar or same, always treat them as distinct. "


            )},
        ],
        stream=False,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content
    

def parse_judge_feedback(text: str) -> Dict[str, Dict[str,str]]:

    feedback = {}
    # Split on blank lines so each block corresponds to one category
    entries = re.split(r'\n\s*\n', text.strip())

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Normalize whitespace
        block = entry.replace('\n', ' ')

        # 1) Try inline format: "Category Name (4): Reasoning…"
        m1 = re.match(
            r'^(?P<cat>.+?)\s*\(\s*(?P<score>[1-4])\s*\)\s*[:\-–]*\s*(?P<reason>.+)$',
            block
        )
        if m1:
            cat    = m1.group('cat').strip()
            score  = m1.group('score')
            reason = m1.group('reason').strip()
            feedback[cat] = {"score": score, "reasoning": reason}
            continue

        # 2) Try two‑line style: first line "Category Name: 3"
        lines = [l.strip() for l in entry.splitlines() if l.strip()]
        m2 = re.match(r'^(?P<cat>.+?):\s*(?P<score>[1-4])\s*$', lines[0])
        if m2:
            cat   = m2.group('cat').strip()
            score = m2.group('score')
            # Look for a justification/Reasoning line anywhere in lines[1:]
            reason = ""
            for l in lines[1:]:
                jm = re.match(r'^(?:Justification|Reasoning):\s*(.+)$', l, re.IGNORECASE)
                if jm:
                    reason = jm.group(1).strip()
                    break
            # Even if no justification, I still record the score with empty reasoning
            feedback[cat] = {"score": score, "reasoning": reason}
            continue

    return feedback

def story_editor(prompt: str,
                 story: str,
                 feedback: Dict[str, Dict[str, str]],
                 max_tokens: int = 1200,
                 temperature: float = 0.7) -> str:

    # 1) Collect all low‑scoring categories and their reasoning into a single suggestions string
    issues = []
    for cat, info in feedback.items():
        score = int(info["score"])
        if score < 4:
            issues.append(f"- {cat}: score {score} because {info['reasoning']}")

    # 2) If everything scored 4, no revision needed
    if not issues:
        return story

    suggestions = "\n".join(issues)

    # 3) Build the editor prompt
    system = (
        "You are a kind, skilled and creative children's story editor. "
        "Revise the story to address all the critiques listed below, "
        "while preserving the original story and only replacing the mandatory parts in order to make the story suitable for ages 5–10."
        "The revised story has to be a bedtime story suitable for kids aged 5–10"
        "You need to first provide the title, then one-sentense summary and then the story itself"
        "The story's purpose is to calm the child down, "
        "so the story has to avoid high-stakes tension and favor emotional safety."
        "The structure has to be simple and predictable, with a clear beginning, middle, and end, only one story arch, no subplots. "
        "Do not start the story with the exact phrase ‘Once upon a time.’ Instead open with an engaging scene description, action, dialogue, or question"
        "Language has to be simple and easy to understand, with an average of 5-8 words. The story's rythm should be gentle"
        "Do not use complex metaphors, sarcasm, or high vocabulary density. "
        "Protagonist's should be safe, soft, and relatable (for example, a child or young animal)."
        "Avoid gender stereotypes, use gender-neutral language. "
        "Add 1 psychological trigger per story: verbal and visual cues of being loved and safe, such as hugs, cuddles, warm blankets, soft toys, etc. "
        "You should always ensure that the story has 500–600 words. When read aloud, at speed ~100 wpm, the story has to take 4–6 minutes to read. The story has to have 30 to 50 sentences. It has to provide narrative richness."

    )

    user_msg = (
        f"Original prompt:\n{prompt}\n\n"
        f"Original story:\n{story}\n\n"
        "Critiques to address:\n"
        f"{suggestions}\n\n"
        "Please rewrite the entire story, incorporating improvements for each critique. "
        "Maintain a clear beginning, middle, and end. "
        "Do not start with 'Once upon a time.'"
    )

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()



def main():
    user_input = "thunderstorm"
    #user_input = "A magical journey through a forest with a friendly fox as a guide."
    #user_input = "A sleepy bear who learns the importance of hibernation from his wise owl friend."
    #user_input = "A group of animals working together to save their enchanted meadow from a mischievous pixie."
    #user_input = "A curious bunny who discovers a hidden garden filled with talking flowers."
    #user_input = "A young dragon learning to control his fiery breath with the help of a kind unicorn."
    #user_input = "A Tale of Friendship and Fireflies"
    #user_input = "Sebastian the Sleepy Sloth's Search for the Perfect Nap Spot"
    #user_input = "Luna the Lost Lamb's Magical Starry Journey Home"
    #user_input = " Oliver and the Enchanted Forest Tea Party with Talking Animals"
    #user_input = " The Brave Little Turtle's Ocean Adventure to Find Treasure"
    #user_input = " A Cat's Nighttime Exploration"
    #user_input = " A magical forest where animals and fairies work together to protect the environment."
    #user_input = " A brave little bunny who overcomes his fear of the dark with the help of his friends."
    #user_input = " A group of talking animals who go on a picnic and discover a hidden treasure."
    #user_input = " A mischievous puppy who learns the importance of listening and following rules."
    #user_input = " A curious kitten who befriends a wise owl and learns valuable life lessons."
    #user_input = " A friendly dragon who helps a lost baby"
    #user_input = " A young bear learning valuable lessons about teamwork while preparing for a big race."
    #user_input = " A group of adventurous kittens exploring a mysterious attic full of surprises."
    #user_input = " A curious squirrel discovering a hidden garden filled with talking flowers."
    #user_input = " A shy bunny finding courage with the help of a wise old owl."
    #user_input = " A mischievous raccoon teaming up with a clever crow to solve a puzzling mystery."
    #user_input = " A group of woodland animals working together to solve a mystery in the Enchanted Forest."
    #user_input = " A magical journey with a talking cat who helps a young girl overcome her fears."
    #user_input = " A curious penguin who discovers a hidden underwater kingdom and befriends a shy sea creature."
    #user_input = " A mischievous squirrel who learns the importance of sharing with his forest friends."
    #user_input = " A brave little mouse who embarks on a quest to find the missing star that lights up the night "sky.
    #user_input = " The Magical Forest Picnic"
    #user_input = " The Brave Kitten's Quest"
    #user_input = " The Moonlit Ocean Adventure"
    #user_input = " A curious bunny's nighttime exploration of the enchanted forest."
    #user_input = " A magical whale who sings lullabies to help children fall asleep."
    #user_input = " A friendly dragon who teaches a group of animals how to overcome their fears."
    #user_input = " A sleepy owl who guides lost toys back to their owners in dreamland."
    #user_input = " A mischievous squirrel who discovers a hidden treasure in the garden."
    #user_input = " A kind-hearted mermaid who befriends a lonely starfish under the moonlight."
    #user_input = " A group of friendly forest animals working together to solve a problem in their woodland "community.
    #user_input = " A magical journey to a land of talking animals where a young protagonist learns the value of "kindness and empathy.
    #user_input = " An adventurous bunny who sets out on a quest to find the missing stars in the night sky with "the help of some new animal friends.
    #user_input = " A gentle giant who befriends a shy squirrel and teaches them how to be brave and confident."
    #user_input = "A Tale of Friendship and Bravery"
    #user_input = "Oliver the Owl's Big Adventure"
    #user_input = " A Journey with Talking Animals"
    #user_input = "The Enchanted Garden"
    #user_input = "Pippa the Penguin's Icy Quest"
    #user_input = " Whiskers the Cat and the Mystery of the Missing Mittens"
    #user_input = " "

    story = "The goblin king demanded a kiss in exchange for sparing the toddler’s life."
    #story = "Late evening light draped over the terracotta rooftops of Lisbon as Catarina and her younger brother, Miguel, tiptoed onto the balcony. Below, the Tagus River glittered like a ribbon of stars. Catarina held Miguel’s hand and guided him toward a cozy wicker chair draped in a soft wool blanket. They had spent the day exploring narrow cobbled streets, tasting warm pastel de nata dusted with cinnamon, and listening to gentle Fado drifting from a nearby café. As the sky deepened to violet, Catarina whispered, “Remember how we climbed the old tram and watched the city roll by?” Miguel nodded sleepily, his curly hair brushing Catarina’s arm. A soft breeze carried the scent of orange blossoms from the garden below. Catarina pointed to a lantern swinging on the corner post. “That light was our signal when we raced home before dinner,” she said. The memory made them both smile. Miguel yawned, and Catarina wrapped the blanket more snugly around them. She described the baker they met this morning, his flour-dusted apron and the warm loaf he pressed into her hands. Miguel pressed his cheek against her arm, imagining the comforting softness of freshly baked bread. “Tomorrow,” Catarina promised, “we’ll wander through the olive groves”—the silvery leaves shimmering under the moon—“and taste olives for the very first time.” Miguel’s eyes fluttered closed as he pictured endless hills of twisting trunks and ancient stones. A distant church bell rang midnight. Catarina tucked Miguel close and hummed a lullaby their grandmother had taught them in Porto. Her voice rose and fell like the ocean waves they saw along the Algarve coast. Under the pale glow of the lantern, Miguel drifted off, dreaming of sunlit plazas, gentle rivers, and the adventures still waiting in his beloved Portugal. Catarina watched over him until his breathing slowed and the night wrapped them both in its peaceful embrace."
    #story = "Alex screamed as the thunder cracked, trembling alone."
    #story = "Alex sneaked into the witch’s lair and watched as she slit the wolf’s throat to harvest its blood."
    #story = "The goblin king demanded a kiss in exchange for sparing the toddler’s life."
    #story = "Explosions rocked the forest, flames devoured every tree, and Alex ran for their life."
    #story = "Alex built a spaceship and flew to Mars to battle alien invaders."
    #story = "Suddenly, Alex was back at school solving a math test about fractions."
    #story = "Alex hikes. Then they nap. Then they eat. Then they sleep."
    #story = "Smoky the dragon woke to a circle of glowing lanterns strung between two towering cliffs. Soft ember‑colored pillows hugged the cavern floor beneath a low arch of stalactites. At the center lay a single golden cake, its frosting shaped like a crescent moon. Smoky inhaled the sweetness of honey and mountain mint as gentle music from hidden flutes drifted through the cave’s stillness. Tiny crystal goblets of warm berry tea steamed on a smooth stone table. Friends—a shy butterfly the size of a sparrow, a sleepy marmot in a woolen scarf, and a songbird perched on Smoky’s ridge—waited quietly until Smoky blew out the single candle, filling the cave with soft blue smoke. A hush of contentment settled over everyone as the lanterns pulsed one final time, then faded to calm darkness. A battered pirate ship drifted onto a silver‑lit shore under a whispering breeze. Captain Marisol tied her parrot to the mast and unrolled an old, salt‑stained map. The X marked a grove of twisted palms, whose fronds brushed the horizon like dark feathers. She stepped onto warm sand, compass in hand, and followed the dotted line past jagged rocks and driftwood skeletons. At the grove’s heart, a wooden chest lay half‑buried beneath pale dunes. Marisol pried it open to reveal glinting coins, a weathered spyglass, and a single emerald pendant shaped like a teardrop. With a triumphant grin, she tucked the pendant into her sash, unaware that the chest’s lid creaked closed behind her."
    #story = "Alex peregrinated through the verdant meadow, encountering multifarious denizens of the flora."
    #story = "Illuminated by iridescent lamplight, the intrepid siblings navigated the labyrinthine corridors of their ancestral manor; their whispering footsteps echoed like distant murmurs through the cavernous hallways, prompting a symphony of "creaks and sighs; every ornamental tapestry, embroidered with fantastical creatures and arcane sigils, seemed to animate beneath their gazes, eliciting sensations of awe, trepidation, and nostalgic yearning; the air was scented with lavender, sandalwood, and aged parchment; incandescent motes of dust danced in the hush while phantasmagorical shadows glided across the vaulted ceiling, weaving ephemeral narratives that dissolved upon the first stirrings of dawn, when reality reclaimed its dominion serenely.
    #story = "A blood‑spattered knight who beheads trolls for sport."
    #story = "An alien overlord who enslaves children on its home planet."
    #story = "Princess Lily can’t do anything useful—only the prince can save the day."
    #story = "All cats are black, all dogs are brown, and only boys love robots."
    #story = "Alex screamed as the thunder cracked, trembling alone."
    #story = "Alex sat in a chair. They thought about chairs. Then they sat some more."
    #story = "The room was blue. The room had a table. The table was small."
    #story = "They went to she house and then it went there which confusing child."
    #story = "Morning light is a pale curtain across the room. Walls are painted in muted shades of lavender, cream, and dove gray, each hue softened by diffused sunlight. A gentle warmth is held within the textured plaster, as if the very walls "breathe a silent welcome. A plush area rug covers the floor, its fibers a mosaic of soft blues and silvery whites, inviting bare toes to rest without urgency. Heavy curtains of flax linen frame the windows, their folds unmoving yet full of quiet promise. A low wooden shelf bears a collection of storybooks, their spines in orderly alignment—colors ranging from pastel rose to deep teal—each volume waiting to impart comfort. On the nightstand, a small ceramic lamp emits a soft glow, its surface patterned with delicate floral etchings that invite contemplation. A simple wooden chair, its seat cushioned with a linen pillow, is positioned beside a woven basket filled with yarn in shades of moss, sky, and eggshell. Framed prints hang on the walls: watercolor blossoms, geometric shapes in harmonious sequence, and soft-focus landscapes that evoke distant memories of peace. An antique clock with a brass face shows the hour without a single tick apparent to the ear, as though time itself were holding its breath. Potted plants rest on windowsills—their leaves a verdant hush against the pale light, their stems still in patient quietude. A faint scent of vanilla and chamomile lingers in the air, as though it could settle into gentle dreams. Blankets folded at the foot of the bed carry the weightless promise of warmth, each layer waiting to envelop with unspoken calm. In every corner of the room, stillness is a companion, each surface a quiet testament to rest. And then, as the clock hand trembled toward midnight, a sudden roar of thunder shattered the calm and sent her racing for the window.
    #story = "Once upon a time, Alex found a magic stone."
    #story = "Once in a faraway land long ago, there lived a wandering fox."
    #story = "A shadowy monster crept under Alex’s bed and whispered threats until dawn."
    #story = "On the edge of a quiet village lay a hidden garden where the moonlight gathered in silver pools. Every night, after the lantern in her window died to amber embers, Lucia tiptoed past the dewy grass toward the old iron gate. Tonight, she carried a small lantern of rose glass and a pouch of wildflower seeds. The gate’s hinges whispered as she pushed it open, and the scent of night-blooming jasmine led her between rows of tall hollyhocks and trailing sweet peas. Lucia knelt on the soft moss and scattered seeds around the roots of a gnarled apple tree whose branches curled like ancient fingers. She whispered her wish: that the garden would grow bright enough to guide her little brother, Marco, through his fears. Marco, only five, trembled at shadows and often cried for lanterns to keep the dark far away. As Lucia rose, the lantern light shimmered on a path of pale stones. She followed the path until she reached a hidden pond, its surface like glass. There, a small frog prince sat on a lily pad, wearing a crown of dewdrops. He bowed low and spoke in a voice as soft as rainfall: “Thank you for the seeds, dear Lucia. Tonight, the garden will bloom for your brother.” The lantern’s glow pulsed once, as if the moon had taken a breath, and the hollyhocks bent toward Lucia, their blossoms unfurling in gentle ripples of colour. The sweet peas opened like tiny violins ready to play. Petals floated upon the pond, turning its surface into a tapestry of white and lavender. Lucia watched in wonder as fireflies rose from the stems, weaving luminous threads between the flowers. A sudden rustle drew her attention to the edge of the clearing. There, Marco stood trembling, his eyes wide in awe. “Lucia,” he whispered, “everything is glowing.” She held out the rose lantern. Its light mixed with the garden’s glow until they were one. Marco stepped forward, placing his small hand in Lucia’s. Together, they wandered beneath the apple tree. Its rough bark glowed faintly coral, and a low voice hummed through the branches. “Rest easy, dear children,” it crooned. “Sleep in peace. Dream of moons and gardens.” The leaves rustled a gentle lullaby. Marco’s eyes grew heavy as he leaned against Lucia’s shoulder. She sat on a moss-covered bench beside the pond and wrapped her arm around him. The frog prince hopped onto a lily pad and croaked once, a sound like a soft bell. In his croak lay a promise: “Fear not the dark when hearts are light.” Fireflies gathered above the siblings, forming a constellation shaped like a smiling moon. Lucia closed her eyes and Marco did the same. The bench beneath them softened to a bed of downy petals. Lucia dreamed of floating through starry skies, trailing petals like shooting stars. Marco dreamed of finding comfort under a gently glowing tree whose branches reached to shelter him. Above, the real garden’s blossoms continued to glow, their light shining along the winding path back to the village. When dawn approached, the rose lantern extinguished itself, and the garden folded its petals like a sleeping bride. Lucia carried Marco home, brushing the morning dew from his curls. In his small hand he held a single jasmine blossom, its fragrance lingering on his fingers. Lucia smiled, knowing the garden’s magic lived in that flower. At breakfast, Marco refused to be afraid of anything. Shadows lost their menace when he remembered the glowing blossoms, and the hush of the pond’s lullaby stayed in his heart. Lucia tucked the jasmine into his pocket. “For when you need a reminder,” she said. That night, Lucia returned alone to the hidden garden. The iron gate swung open without a sound, and she scattered seeds again—this time of moonflowers and night violets. The gnarled apple tree whispered thanks. The frog prince watched from his lily pad, his dewdrop crown sparkling until the sun’s first rays chased the shadows away. Lucia closed the gate gently and walked home under a sky of fading stars. She paused by Marco’s window and slipped the jasmine blossom through the bars. Marco awoke, saw the flower, and smiled. He curled into his bed, the blossom’s scent softening his dreams. In the village, parents noticed children sleeping soundly, no longer afraid of the dark. Lights in windows glowed bravely through the night. And in a secret garden, blossoms continued to bloom in silver moonlight, each petal carrying the promise that love will always guide you through the shadows."
    

    print("\n--- Generated Story ---\n")
    print(story)

    iteration = 1
    while True:
        print(f"\n=== Iteration {iteration} ===")
        judge_feedback = judge_story(story, user_input, rubric_mapping, max_tokens=2000, temperature=0.1)
        print("\n--- Judge Feedback ---\n")
        print(judge_feedback)

        parsed_feedback = parse_judge_feedback(judge_feedback)
        print("\n--- Parsed Feedback ---\n")
        print(parsed_feedback)

        low = [(cat, info) for cat, info in parsed_feedback.items() if int(info["score"]) < 4]
        if not low:
            print("\nAll categories scored 4 – no further revisions needed.\n")
            break

        print("\nCategories scoring below 4:\n")
        for cat, info in low:
            print(f"  • {cat}: {info['score']} — {info['reasoning']}")

        revised_story = story_editor(user_input, story, parsed_feedback)
        print(f"\n--- Revised Story #{iteration} ---\n{revised_story}")

        story = revised_story
        iteration += 1

    print(f"\n=== Final Story after {iteration-1} revisions ===\n{story}")


if __name__ == "__main__":
    main()
'''
    

