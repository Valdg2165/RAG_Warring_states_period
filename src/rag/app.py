"""
app.py  —  Gradio GUI for the Warring States KGE-RAG
=====================================================

Launches a web interface at http://localhost:7860

Usage
-----
    python src/rag/app.py
    python src/rag/app.py --model mistral
    python src/rag/app.py --share          # public Gradio link
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # fix Anaconda OpenMP conflict on Windows

import argparse
import sys
from pathlib import Path

# Make sure project root is on PYTHONPATH so we can import rag_kge
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import gradio as gr
from src.rag.rag_kge import (
    load_resources,
    answer_question,
    answer_baseline,
    run_evaluation,
    DEFAULT_MODEL,
    TTL_PATH,
    KGE_DIR,
)

# ── Global state (loaded once at startup) ─────────────────────────────────────

RESOURCES = None


def _ensure_loaded():
    global RESOURCES
    if RESOURCES is None:
        RESOURCES = load_resources(TTL_PATH, KGE_DIR)
    return RESOURCES


# ── Handlers ──────────────────────────────────────────────────────────────────

def handle_question(question: str, model: str, show_context: bool):
    """Run both baseline and RAG, return all outputs."""
    if not question.strip():
        return "", "", "", ""

    resources = _ensure_loaded()
    q = question.strip()

    baseline = answer_baseline(q, model)
    result   = answer_question(q, resources, model)

    detected = ", ".join(result["detected_entities"]) or "(none)"
    similar  = ", ".join(result["similar_entities"][:6]) or "(none)"
    triples  = result["context_triples"]

    meta = (
        f"**Detected entities** ({len(result['detected_entities'])}):\n{detected}\n\n"
        f"**KGE-similar entities** ({len(result['similar_entities'])}):\n{similar}\n\n"
        f"**Context triples used**: {triples}"
    )

    context_text = result["context"] if show_context else "(enable 'Show context triples' above)"

    return baseline, result["answer"], meta, context_text


def handle_eval(model: str):
    resources = _ensure_loaded()
    records   = run_evaluation(resources, model)

    rows = []
    for r in records:
        tag = "✅" if r["correct"] else "❌"
        rows.append(
            f"| {tag} | {r['question'][:50]} | {r['context_triples']} | "
            f"{', '.join(r['detected_entities'][:3]) or '—'} |"
        )

    score  = sum(r["correct"] for r in records)
    header = (
        f"## Evaluation — {score}/{len(records)} correct\n\n"
        "| | Question | Triples | Detected entities |\n"
        "|---|---|---|---|\n"
    )
    return header + "\n".join(rows)


# ── Example questions ─────────────────────────────────────────────────────────

EXAMPLES = [
    ["Who were the students of Confucius?"],
    ["Which states were at war with the State of Qin?"],
    ["Who is an intellectual descendant of Confucius?"],
    ["Who was born in the State of Zhao?"],
    ["Which philosophers influenced Epicharmus of Kos?"],
    ["Which people held a position as writer?"],
    ["What did Laozi author?"],
    ["Who did Aristotle influence?"],
]


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
footer { display: none !important; }
.answer-col textarea { font-size: 15px; line-height: 1.6; }
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Warring States KG — RAG Demo",
        theme=gr.themes.Soft(),
        css=CSS,
    ) as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🏯 Warring States Knowledge Graph — RAG
            Ask questions about the **Warring States period** of ancient China.
            Compare the raw LLM answer against the **KGE-grounded RAG** answer.
            """
        )

        # ── Settings ──────────────────────────────────────────────────────────
        with gr.Row():
            model_dd = gr.Dropdown(
                choices=["gemma:2b", "mistral", "tinyllama", "llama3", "phi3:mini"],
                value=DEFAULT_MODEL,
                label="Ollama model",
                scale=1,
            )
            show_ctx = gr.Checkbox(
                label="Show context triples", value=False, scale=1
            )

        # ── Question ──────────────────────────────────────────────────────────
        question_box = gr.Textbox(
            placeholder="e.g. Who were the students of Confucius?",
            label="Your question",
            lines=2,
        )
        with gr.Row():
            submit_btn = gr.Button("Ask", variant="primary")
            gr.ClearButton([question_box], value="Clear")

        # ── Side-by-side answers ──────────────────────────────────────────────
        with gr.Row():
            with gr.Column(elem_classes="answer-col"):
                gr.Markdown("### 🤖 LLM only (no knowledge graph)")
                baseline_box = gr.Textbox(
                    label="Baseline answer",
                    lines=6,
                    interactive=False,
                )

            with gr.Column(elem_classes="answer-col"):
                gr.Markdown("### 📚 RAG answer (KG + LLM)")
                rag_box = gr.Textbox(
                    label="KGE-RAG answer",
                    lines=6,
                    interactive=False,
                )

        # ── Retrieval details ─────────────────────────────────────────────────
        meta_box = gr.Markdown()

        # ── Context triples ───────────────────────────────────────────────────
        context_box = gr.Textbox(
            label="Context triples passed to LLM",
            lines=10,
            interactive=False,
        )

        # ── Examples ──────────────────────────────────────────────────────────
        gr.Markdown("### Example questions")
        gr.Examples(examples=EXAMPLES, inputs=question_box, label="")

        # ── Evaluation ────────────────────────────────────────────────────────
        with gr.Accordion("Run full evaluation (7 questions)", open=False):
            eval_btn    = gr.Button("Run evaluation", variant="secondary")
            eval_output = gr.Markdown()

        # ── Events ────────────────────────────────────────────────────────────
        outputs = [baseline_box, rag_box, meta_box, context_box]

        submit_btn.click(
            fn=handle_question,
            inputs=[question_box, model_dd, show_ctx],
            outputs=outputs,
        )
        question_box.submit(
            fn=handle_question,
            inputs=[question_box, model_dd, show_ctx],
            outputs=outputs,
        )
        eval_btn.click(
            fn=handle_eval,
            inputs=[model_dd],
            outputs=[eval_output],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Gradio GUI for KGE-RAG")
    ap.add_argument("--model",  default=DEFAULT_MODEL, help="Default Ollama model")
    ap.add_argument("--port",   type=int, default=7860, help="Port to serve on")
    ap.add_argument("--share",  action="store_true",   help="Create a public Gradio link")
    args = ap.parse_args()

    print("Pre-loading knowledge graph and embeddings …")
    _ensure_loaded()

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
