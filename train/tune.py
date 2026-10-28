#!/usr/bin/env python3
"""The Tinker arm: fine-tune a small open model to do one job — turn a family transcript into claims.

    python train/tune.py probe                 # is the key alive and the plumbing right (1 item)
    python train/tune.py train                 # LoRA SFT on data/extract/train.jsonl
    python train/tune.py score base            # score the held-out set with the base model
    python train/tune.py score tuned           # ... and with the adapter
    python train/tune.py report                # side-by-side table for the post

Why fine-tune at all: the base model writes fluent Vietnamese and then invents. The job here is narrower
than conversation — copy a span of the transcript into a structured claim, or emit nothing. That is a
behaviour a small adapter can learn from a handful of examples, and the point of the scoring is to find out
whether it actually did, on passages it has never seen. If it does not, the losing run is published.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import tinker
from tinker import types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data" / "extract"
OUT = ROOT / "data" / "eval"
STATE = HERE / "state.json"
BASE_MODEL = os.environ.get("TUNE_BASE", "Qwen/Qwen3.5-4B")

sys.path.insert(0, str(ROOT))
from core import extract, textnorm, guard   # noqa: E402

SYSTEM = (
    "Bạn đọc một đoạn ghi âm của người lớn tuổi trong gia đình Việt Nam và rút ra những điều gia đình "
    "cần biết về ngày giỗ.\n"
    "Chỉ được chép NGUYÊN VĂN từ đoạn ghi âm vào trường \"quote\". Không diễn giải, không thêm, không bịa.\n"
    "Nếu đoạn ghi âm không nói rõ, bỏ qua.\n"
    "Trả về JSON: {\"claims\": [{\"kind\": \"person|gio_date|offering|instruction\", "
    "\"quote\": \"...\", \"lunar_day\": null, \"lunar_month\": null, \"leap\": false}]}"
)


def load(split: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / f"{split}.jsonl").open(encoding="utf-8")]


def user_text(row: dict) -> str:
    lines = [f"[{int(s)//60:02d}:{int(s)%60:02d}] {t}" for s, _e, t in row["pieces"]]
    return "BẢN GHI:\n" + "\n".join(lines)


def target_json(row: dict) -> str:
    claims = [{"kind": g["kind"], "quote": g["quote"],
               "lunar_day": g.get("lunar_day"), "lunar_month": g.get("lunar_month"),
               "leap": False} for g in row["gold"]]
    return json.dumps({"claims": claims}, ensure_ascii=False)


def _tokenizer(base_model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


def render_prompt(tok, row: dict) -> str:
    """The exact prompt both arms see. `enable_thinking=False` is not decoration.

    Qwen3.5 is a reasoning model: given a normal generation prompt it emits "Thinking Process: ..." before
    any answer, so the JSON never appears where a strict parser looks for it and the tuned arm would look
    worse than it is for a reason that has nothing to do with fine-tuning. Both arms are rendered through the
    same template so the only difference between them is the adapter.
    """
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(row)}]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_data(rows: list[dict], tok) -> list[tuple[list[int], list[int], list[float]]]:
    """(x, y, w) triples where w is aligned with y, not with x — weights sit on the completion tokens."""
    out = []
    for row in rows:
        prompt_ids = tok.encode(render_prompt(tok, row), add_special_tokens=False)
        completion_ids = tok.encode(target_json(row) + "<|im_end|>", add_special_tokens=False)
        full = prompt_ids + completion_ids
        w = [0.0] * len(prompt_ids) + [1.0] * len(completion_ids)
        out.append((full[:-1], full[1:], w[1:]))
    return out


# ------------------------------------------------------------------ scoring

def claims_match(model_claim: dict, gold: dict, overlap: float = 0.6) -> bool:
    """One matching rule for every arm, so the table compares like with like.

    A date matches when the lunar day and month match — the quote is the citation, the date is the answer, and
    a model that quotes the whole sentence around the date is not wrong. A text claim matches when one
    normalised quote contains the other, or when at least `overlap` of the gold's syllables appear in the
    model's quote, which accepts a longer verbatim span without accepting a paraphrase.
    """
    if model_claim["kind"] != gold["kind"]:
        return False
    if gold["kind"] == "gio_date" and gold.get("lunar_day") is not None:
        return (model_claim.get("lunar_day") == gold.get("lunar_day")
                and model_claim.get("lunar_month") == gold.get("lunar_month"))
    a, b = textnorm.normalise(model_claim["quote"]), textnorm.normalise(gold["quote"])
    if not a or not b:
        return False
    if b in a or a in b:
        return True
    gold_syl = set(b.split())
    return bool(gold_syl) and len(gold_syl & set(a.split())) / len(gold_syl) >= overlap


def score_output(text: str, row: dict) -> dict:
    """Precision/recall of the claims, after the same guard the product uses."""
    claims = extract.parse_model_output(text)
    pieces = [(s, e, t) for s, e, t in row["pieces"]]
    today = (2026, 10, 28)
    verdict = guard.review(claims, pieces, today)
    kept = [{"kind": c.kind, "quote": c.quote, "lunar_day": c.lunar_day,
             "lunar_month": c.lunar_month} for c in verdict.kept]
    gold = list(row["gold"])
    used: set[int] = set()
    hits = 0
    for gc in kept:
        for j, g in enumerate(gold):
            if j in used:
                continue
            if claims_match(gc, g):
                used.add(j)
                hits += 1
                break
    return {
        "kept": len(kept), "gold": len(gold), "hit": hits,
        "precision": hits / len(kept) if kept else 0.0,
        "recall": hits / len(gold) if gold else 0.0,
        "refused": len(verdict.refused), "uncited": verdict.uncited_kept,
        "parsed": len(claims),
    }


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def summarise(rows: list[dict]) -> dict:
    n = len(rows) or 1
    p = sum(r["precision"] for r in rows) / n
    rec = sum(r["recall"] for r in rows) / n
    return {"n": len(rows), "precision": round(p, 4), "recall": round(rec, 4),
            "f1": round(f1(p, rec), 4),
            "uncited": sum(r["uncited"] for r in rows),
            "refused": sum(r["refused"] for r in rows),
            "mean_latency_s": round(sum(r.get("latency_s", 0) for r in rows) / n, 2)}


# ------------------------------------------------------------------ commands

def _sampling_client(arm: str):
    sc = tinker.ServiceClient()
    if arm == "base":
        return sc.create_sampling_client(base_model=BASE_MODEL)
    st = json.loads(STATE.read_text())
    handle = st["tuned"]
    if isinstance(handle, str) and handle.startswith("tinker://"):
        return sc.create_sampling_client(base_model=st.get("base_model", BASE_MODEL), model_path=handle)
    return tinker.SamplingClient.from_sampler_handle(handle, api_key=os.environ["TINKER_API_KEY"])


def _ask(cli, row: dict) -> tuple[str, float]:
    ids = _TOK.encode(render_prompt(_TOK, row), add_special_tokens=False)
    t0 = time.time()
    fut = cli.sample(prompt=types.ModelInput.from_ints(ids), num_samples=1,
                     sampling_params=types.SamplingParams(max_tokens=int(os.environ.get("MAX_TOKENS", "400")),
                                                          temperature=0.0))
    res = fut.result() if hasattr(fut, "result") else fut
    return _TOK.decode(res.sequences[0].tokens, skip_special_tokens=True), time.time() - t0


def probe() -> int:
    row = load("heldout")[0]
    cli = _sampling_client("base")
    text, dt = _ask(cli, row)
    m = score_output(text, row)
    print(f"model={BASE_MODEL} latency={dt:.1f}s kept={m['kept']} gold={m['gold']} hit={m['hit']}")
    print("raw:", text[:500])
    return 0


def train(steps: int = 80, lr: float = 1e-4, rank: int = 16, batch: int = 4) -> int:
    import math
    import random

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=rank,
                                        user_metadata={"project": "so-gio", "task": "claims-from-transcript"})
    tok = tc.get_tokenizer()
    data = build_data(load("train"), tok)
    print(f"training {BASE_MODEL} rank={rank} lr={lr} on {len(data)} passages, {steps} steps", flush=True)
    rng = random.Random(0)
    order = list(range(len(data)))
    per_epoch = math.ceil(len(data) / batch)
    losses: list[float] = []
    for step in range(steps):
        if step % per_epoch == 0:
            rng.shuffle(order)
        idxs = [order[(step * batch + k) % len(order)] for k in range(batch)]
        batch_data, n_sup = [], 0
        for i in idxs:
            x, y, w = data[i]
            batch_data.append(types.Datum(
                model_input=types.ModelInput.from_ints(x),
                loss_fn_inputs={"target_tokens": types.TensorData(data=y, dtype="int64"),
                                "weights": types.TensorData(data=w, dtype="float32")}))
            n_sup += sum(w)
        fb = tc.forward_backward(batch_data, loss_fn="cross_entropy").result()
        tc.optim_step(adam_params=types.AdamParams(learning_rate=lr)).result()
        raw = fb.metrics.get("loss:sum", float("nan"))
        losses.append(raw / n_sup if n_sup else raw)
        if step % 10 == 0 or step == steps - 1:
            tail = losses[-10:]
            print(f"  step {step:3d}  loss/supervised-token={sum(tail)/len(tail):.4f}  "
                  f"supervised_tokens={n_sup}", flush=True)
    cli = tc.save_weights_and_get_sampling_client()
    handle = cli.get_sampler_handle()
    STATE.write_text(json.dumps({
        "tuned": handle if isinstance(handle, str) else repr(handle),
        "base_model": BASE_MODEL, "steps": steps, "rank": rank, "lr": lr,
        "rows": len(data), "final_loss": sum(losses[-10:]) / max(len(losses[-10:]), 1),
    }, indent=2))
    print(f"saved handle -> {STATE.name}")
    return 0


def score(arm: str) -> int:
    cli = _sampling_client(arm)
    out = []
    for row in load("heldout"):
        text, dt = _ask(cli, row)
        m = score_output(text, row)
        m.update(latency_s=round(dt, 2), id=row["id"], raw=text[:800])
        out.append(m)
        print(f"  {row['id']} kept={m['kept']:2d} gold={m['gold']:2d} hit={m['hit']:2d} "
              f"p={m['precision']:.2f} r={m['recall']:.2f} {dt:5.1f}s")
    summary = summarise(out)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"tinker_{arm}.json"
    path.write_text(json.dumps({"arm": arm, "base_model": BASE_MODEL, "summary": summary,
                                "items": out}, ensure_ascii=False, indent=2))
    print(f"\n{arm}: {json.dumps(summary)}")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


def report() -> int:
    rows = {}
    for arm in ("base", "tuned"):
        p = OUT / f"tinker_{arm}.json"
        if p.exists():
            rows[arm] = json.loads(p.read_text())["summary"]
    if not rows:
        sys.exit("no scores yet")
    print(f"{'arm':8s} {'n':>3s} {'P':>6s} {'R':>6s} {'F1':>6s} {'uncited':>8s} {'refused':>8s} {'s/item':>7s}")
    for arm, s in rows.items():
        print(f"{arm:8s} {s['n']:3d} {s['precision']:6.3f} {s['recall']:6.3f} {s['f1']:6.3f} "
              f"{s['uncited']:8d} {s['refused']:8d} {s['mean_latency_s']:7.2f}")
    return 0


def main(argv: list[str]) -> int:
    global _TOK
    if not os.environ.get("TINKER_API_KEY"):
        key_file = pathlib.Path.home() / ".config" / "hf26" / "tinker_api_key"
        if key_file.exists():
            os.environ["TINKER_API_KEY"] = key_file.read_text().strip()
    _TOK = _tokenizer(BASE_MODEL)
    cmd = argv[1] if len(argv) > 1 else "probe"
    if cmd == "probe":
        return probe()
    if cmd == "train":
        return train(steps=int(argv[2]) if len(argv) > 2 else 60)
    if cmd == "score":
        return score(argv[2] if len(argv) > 2 else "base")
    if cmd == "report":
        return report()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
