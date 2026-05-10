    #context.py

def build_context(chunks):

    context_lines = []

    for chunk in chunks:

        context_lines.append(
            f"""
SOURCE: {chunk['source']}
PAGE: {chunk['page']}
PARAGRAPH: {chunk['paragraph']}

CONTENT:
{chunk['text']}
"""
        )

    return "\n".join(context_lines)