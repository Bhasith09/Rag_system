def build_context(chunks):
    context_lines = []

    for i, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page", 0) + 1
        paragraph = chunk.metadata.get("paragraph", 1)

        context_lines.append(
            f"[{i}] (Page {page}, Paragraph {paragraph})\n"
            f"{chunk.page_content}"
        )

    return "\n\n".join(context_lines)