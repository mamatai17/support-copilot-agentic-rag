from __future__ import annotations
from pathlib import Path
import urllib.request

OUT_DIR = Path("data/raw/kb_k8s_md")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# A small, useful subset of K8s docs (raw markdown in kubernetes/website repo)
FILES = [
    ("overview.md", "https://raw.githubusercontent.com/kubernetes/website/master/content/en/docs/concepts/overview/what-is-kubernetes.md"),
    ("objects.md", "https://raw.githubusercontent.com/kubernetes/website/master/content/en/docs/concepts/overview/working-with-objects/object-management.md"),
    ("deployments.md", "https://raw.githubusercontent.com/kubernetes/website/master/content/en/docs/concepts/workloads/controllers/deployment.md"),
    ("services.md", "https://raw.githubusercontent.com/kubernetes/website/master/content/en/docs/concepts/services-networking/service.md"),
    ("configmaps.md", "https://raw.githubusercontent.com/kubernetes/website/master/content/en/docs/concepts/configuration/configmap.md"),
]

def fetch(url: str) -> str:
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8", errors="replace")

def main():
    for name, url in FILES:
        text = fetch(url)
        (OUT_DIR / name).write_text(text, encoding="utf-8")
        print("saved", name)
    print(f"KB markdown saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
