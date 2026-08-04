def build_context(chunks):
    context_lines=[]
    for i,chunk in enumerate(chunks,start=1):
        context_lines.append(f"[{i}] SOURCE:\n{chunk}")
    return "\n".join(context_lines)