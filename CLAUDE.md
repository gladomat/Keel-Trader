# Memory: MuninnDB (Canonical)

MuninnDB (muninn MCP) is the canonical memory system. Never use local auto memory.

## Session Start — Always
Call muninn_recall with relevant context before beginning any work.
This loads prior context. Vault: default.

## During Every Session
- Save to Muninn continuously — this is a mindset, not a checklist.
- Anything the user shares or that emerges from the work should be saved immediately.
- Do not evaluate whether it is "important enough".
- Do not wait to be asked. When in doubt, save it.

## Tools
- Recall: muninn_recall (vault, context)
- Store: muninn_remember (vault, concept, content)
- Batch: muninn_remember_batch (vault, memories[])
- Guide: muninn_guide — call on first connect to learn best practices
