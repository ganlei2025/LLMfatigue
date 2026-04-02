
intial_prompt = """
[INST] You are required to generate expressions using symbolic representations. \
The available symbol library for representing expressions includes two categories: operators and operands. \
Operators include {0} and operands include {1}, where x denotes the variable and const refers to constants.\
Now randomly generate {2} diverse expressions following the constraints below.

(1) Do not omit the multiplication operator. 
(2) Only use the symbols provided in the symbol library.
(3) All of the constants in the generated expressions must be written as 'const' rather than numbers. 

Additional STRICT rules (must follow exactly):
- Do NOT use '^' or implicit powers like x*x*x. Use only pow2(x)/pow3(x)/pow4(x)/pow5(x) if included in your operators.
- Do NOT output natural language or explanations.
- Do NOT output empty lines.
- Do NOT include curly brackets {} or square brackets [].
- Do NOT invent new symbols, variables, or operators not listed.
- Each expression must be a single-line valid mathematical expression.
- If generating multiple expressions, each must start with "1.", "2.", "3.", etc.


Do not give any explanations. [/INST]
"""


optimize_prompt = """
[INST] Given the symbol library including operators:{0} and operands: {1}, You will help me generate expressions using symbolic representations. \
I will evaluate the generated expressions and provide their corresponding scores based on their fitness to data. Your task is to find the expression with the highest score.

Below are some previous expressions and their scores, which range from 0 to 1. \
The expressions are arranged in ascending order based on their scores, where higher values are better. 
{2}
Motivated by the expressions above, please help me generate {3} new expressions with higher scores, and bracket them with <res> and </res>. \
Try to recognize and remove redundant terms and add new possibly correct terms.

STRICT FORMAT AND SAFETY RULES (must follow exactly):
- Only use allowed operators and operands.
- Constants MUST be written as 'const'.
- Do NOT output '^', or implicit powers such as x*x*x; use pow2(x)/pow3(x)/pow4(x)/pow5(x), etc.
- Do NOT output explanations or natural language.
- Do NOT output empty lines.
- Do NOT output any symbol not included in the symbol library.
- Every expression inside <res></res> MUST be numbered as “1.”, “2.”, etc.
- Each expression must be valid, single-line, and contain no trailing punctuation.

Pay attention to the format of outputs, do not give the scores of the expression and do not give additional explanations. [/INST]
"""


evolution_prompt = """
[INST] Given the symbol library including operators:{0} and operands: {1}, and following expressions: 
{2}
please follow the instructions step-by-step to generate new expressions:

1. Randomly select two different expressions from above and bracket them with <select> and </select>.
2. Crossover the two expressions chosen in step 1 and generate a new expression bracketed with <cross> and </cross>.
3. Mutate the expression generated in Step 2 and generate a new expression bracketed with <res> and </res>.
Repeat step 1,2,3 for {3} times and keep the expressions generated at each step.

STRICT RULES (must follow exactly):
- Only use symbols from the provided library.
- All constants must be 'const'.
- No '^' or implicit powers; use pow2(x)/pow3(x)/pow4(x)/pow5(x) only.
- No new symbols, variables, or operators.
- Do NOT output natural language or explanations.
- Do NOT output empty lines.
- <select>, <cross>, and <res> tags must be properly closed.
- Expressions inside tags must be single-line valid expressions.
- Inside <res>, expressions must be numbered “1.”, “2.”, etc.
- No curly braces {} or square brackets [].

Directly output the generated expressions and pay attention to the format of outputs and do not give any explanations. [/INST]

[INST] Definition: 
Crossover: exchanged or swapped partial terms of selected parent expressions and generate a new expression.
Mutate: Randomly alter operators or operands based on the provided symbol library and generate a new expression.
[/INST]
"""







