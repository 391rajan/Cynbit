"""Fix the f-string escape issue in app.py."""
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and fix the problematic line
for i, line in enumerate(lines):
    if 'status_colors.get(status, theme.P' in line and 'text_muted' in line:
        print(f"Found at line {i+1}: {repr(line)}")
        # Replace block: lines i-1 through i+3 (the st.markdown call)
        # Insert a new line BEFORE line i-1 (which is st.markdown() open)
        new_lines = lines[:i-1]
        new_lines.append('        _scolor = status_colors.get(status, theme.P["text_muted"])\n')
        new_lines.append('        st.markdown(\n')
        new_lines.append('            f\'<div style="margin-top:.5rem;font-size:.8rem;color:{_scolor}">\'\n')
        new_lines.append('            f"Status: <strong>{status.upper()}</strong></div>",\n')
        new_lines.append('            unsafe_allow_html=True,\n')
        new_lines.append('        )\n')
        # Skip the original 5 lines (i-1 through i+3)
        new_lines.extend(lines[i+4:])
        with open('app.py', 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("Fixed!")
        break
else:
    print("Pattern not found!")
