# Shell guidelines

- Use `python` (not `python3`) to run scripts; it is on PATH in the sandbox.
- Prefer `grep -rn` for searching and `ls -la` for listing so output includes useful details.
- Pipe long output through `head` to keep it readable.
- Put new code files at the sandbox root unless the user names a folder.
