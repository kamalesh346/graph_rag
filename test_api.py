import subprocess, json, sys

BASE = "http://localhost:8000"

def req(method, path, body=None, timeout=60):
    cmd = ["curl", "-s", "--max-time", str(timeout), BASE + path,
           "-H", "Content-Type: application/json"]
    if body:
        cmd += ["-d", json.dumps(body)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return json.loads(r.stdout)
    except Exception:
        return {"_raw": r.stdout[:300]}

results = []

def check(label, ok, detail=""):
    icon = "OK" if ok else "FAIL"
    print(f"[{icon}] {label}{(' -- ' + detail) if detail else ''}")
    results.append(ok)

# 1 Health
d = req("GET", "/api/health")
check("/api/health",
      d.get("vector_store_available") is True,
      "vector=%s neo4j=%s" % (d.get("vector_store_available"), d.get("neo4j_available")))

# 2 GraphRAG research
d = req("POST", "/api/research", {"query": "murder BNS"})
provs = d.get("provisions", [])
sec_ids = [str(p.get("act","")) + ":" + str(p.get("section","")) for p in provs]
check("/api/research (GraphRAG)", len(provs) > 0,
      "%d provisions %s, %d judgments" % (len(provs), sec_ids, len(d.get("judgments",[]))))

# 3 Plain RAG
d = req("POST", "/api/research/plain", {"query": "dowry death"})
provs = d.get("provisions", [])
sec_ids = [str(p.get("act","")) + ":" + str(p.get("section","")) for p in provs]
check("/api/research/plain", len(provs) > 0, "%d provisions %s" % (len(provs), sec_ids))

# 4 LLM ask — meta query
d = req("POST", "/api/ask", {"query": "what sections are available in the application"}, timeout=90)
ans = d.get("answer", "")
check("/api/ask meta-query", d.get("is_out_of_scope") is False and len(ans) > 50,
      "oos=%s answer_len=%d" % (d.get("is_out_of_scope"), len(ans)))

# 5 LLM ask — specific section
d = req("POST", "/api/ask", {"query": "what is BNS 103"}, timeout=90)
check("/api/ask specific", d.get("is_out_of_scope") is False,
      "oos=%s sources=%s" % (d.get("is_out_of_scope"), d.get("sources")))

# 6 Admin stats
d = req("GET", "/api/admin/stats")
check("/api/admin/stats",
      d.get("statute_sections", 0) == 108 and d.get("judgment_chunks", 0) == 285,
      str(d))

print()
passed = sum(results)
total  = len(results)
print("ALL PASSED (%d/%d)" % (passed, total) if passed == total
      else "SOME FAILED (%d/%d)" % (passed, total))
