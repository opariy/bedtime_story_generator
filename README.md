### Bedtime story generator

## How to Run the Program
There are two modes:
- Final Product Mode => You are asked to provide a topic, and a story will be generated about that topic.
- Testing Mode => Use this mode if you want to test individual functions and see intermediate outputs.

## Running in Final Product Mode:
- Create a file named .env in the project root
- Paste your OpenAI API key into .env using this format: OPENAI_API_KEY={your_key_here}
- From your terminal (or VS Code), run: python main.py
- You will be prompted to enter a topic. The program will then generate and display a bedtime story for ages 5–10.

## Running in Testing Mode
- Open main.py
- Comment out the section marked “FINAL PRODUCT MODE” (so it does not run)
- Uncomment the section marked “DEBUG / TESTING PART”.
   - This block contains sample user_input and story variables, as well as calls to judge_story() and story_editor()
   - You can modify or add your own test cases inside that block to see how the judge and editor functions behave - each step will print intermediate outputs
- Make sure your .env file (with your OpenAI key) is still in place
- Run: python main.py