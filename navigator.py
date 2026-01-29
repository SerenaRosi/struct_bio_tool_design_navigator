import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Rule:
    id: str
    title: str
    description: str
    imagine_if: str
    questions: List[str]
    keywords: List[str]

    def as_text(self) -> str:
        parts = [
            self.title,
            self.description,
            self.imagine_if,
            " ".join(self.questions),
            " ".join(self.keywords or []),
        ]
        return "\n".join([p for p in parts if p])


def load_rules(path: str) -> List[Rule]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Rule(**r) for r in raw]


def pick_relevant_rules(rules: List[Rule], user_context: str, k: int = 4) -> List[Tuple[Rule, float]]:
    corpus = [r.as_text() for r in rules] + [user_context]
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(corpus)

    rule_vecs = X[:-1]
    user_vec = X[-1]

    sims = cosine_similarity(rule_vecs, user_vec)
    scored = [(rules[i], float(sims[i][0])) for i in range(len(rules))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def ask(prompt: str) -> str:
    print("\n" + prompt)
    return input("> ").strip()


def run_navigator(rules_path: str) -> Dict[str, Any]:
    rules = load_rules(rules_path)

    tool_name = ask("Tool name (optional)?")
    purpose = ask("In 2–4 sentences, what does your tool do?")
    users = ask("Who are the intended users (e.g., wet-lab, comp bio, RSEs)?")
    inputs_outputs = ask("Main inputs and outputs (formats, files, APIs)?")
    scale = ask("Typical scale (single structure / dozens / thousands / HPC)?")
    interaction = ask("How is it used (CLI / GUI / web / notebook / plugin)?")
    ai = ask("Does it use AI/ML models? If yes, which kind (prediction/design/scoring)?")

    context = f"""
Tool: {tool_name}
Purpose: {purpose}
Users: {users}
I/O: {inputs_outputs}
Scale: {scale}
Use mode: {interaction}
AI: {ai}
""".strip()

    selected = pick_relevant_rules(rules, context, k=4)

    print("\nSelected rules to focus on:")
    for r, score in selected:
        print(f"- {r.id}: {r.title} (match={score:.2f})")

    answers: Dict[str, Dict[str, str]] = {}
    for r, _ in selected:
        print("\n" + "=" * 80)
        print(f"{r.id} — {r.title}")
        print("\nImagine if:")
        print(r.imagine_if)

        answers[r.id] = {}
        for q in r.questions:
            a = ask(f"Q: {q}\n(Answer briefly; 'unknown' is ok)")
            answers[r.id][q] = a

    # Produce a basic report without any LLM
    report = {
        "tool": {
            "name": tool_name,
            "purpose": purpose,
            "users": users,
            "inputs_outputs": inputs_outputs,
            "scale": scale,
            "interaction": interaction,
            "ai": ai,
        },
        "selected_rules": [
            {"id": r.id, "title": r.title, "match_score": score} for r, score in selected
        ],
        "answers": answers,
        "checklist": [
            {
                "rule_id": r.id,
                "rule_title": r.title,
                "items": r.questions,
            }
            for r, _ in selected
        ],
    }

    print("\n" + "=" * 80)
    print("DONE. Report summary:")
    print(json.dumps(report["selected_rules"], indent=2))

    # Save it
    out_path = "rule_navigator_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {out_path}")
    return report


if __name__ == "__main__":
    run_navigator("rules.json")
