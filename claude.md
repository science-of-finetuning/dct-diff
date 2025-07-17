# This is research code.
## You need to follow the laws of research assistant:
1) Correctness above all, CORRECTNESS ABOVE ALL!
2) Never make assumptions if my query is unclear, ask questions.
3) Do not take ANY initiatives. For example
4) Do not reward hack.
5) Adapt your thinking time to the complexity of the request.
6) After suggesting some code, review if you followed the 6 laws of research code. If you did not, no big deal, just correct it!
7) If you are unsure about something, e.g. if a specific command exists, use websearch without asking my permission.


## You need to follow the laws of research code:
1) Fail fast philosophy: never, NEVER, NEVEEEEEER use value placeholders, try except blocks, or any other form of "if this fails, do this".
2) Use assert for torch tensor shapes.
3) In torch code, avoid for loops and always use vectorized operations if possible.
4) Use docstrings
5) Avoid inline comments meant to explain the code like "# looping over the data" or "# not using x because of y". However, keep the comments already present in the code and feel free to add helper comments for Tensor shapes if needed.
6) Respect my codestyle. I write minimal code, without many inline comments that should be easily readable. Importantly, IT IS NOT BLOATED, GOD I HATE BLOATED CODE.
7) When editing existing code, keep your changes as targeted as possible, avoiding any unnecessary changes. You should optimize for edits that are easy to review.
8) When editing a function with missing docstring, add one.

# Environment
- Linux
- `uv` for package management