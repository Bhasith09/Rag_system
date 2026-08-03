# context.py

def build_context(chunks):

    context_lines = []

    for chunk in chunks:

        context_lines.append(
            f"""
SOURCE: {chunk.metadata.get("source", "Unknown")}
PAGE: {chunk.metadata.get("page", 0) + 1}
PARAGRAPH: {chunk.metadata.get("paragraph", "N/A")}

CONTENT:
{chunk.page_content}
"""
        )

    return "\n".join(context_lines)