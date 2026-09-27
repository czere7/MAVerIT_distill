"""The training side of the distillation, ported from ~/maverit-preference and
~/maverit-finetune so that the harness, the verifier and the trainer live in one project.

    scoring/        the verifier: tiered, repair-free measurement of one generated suite
    quality.py      the rewards that rank rollouts (quality v1 graded, quality_v2's veto)
    chat_template   the qwen3_5_nothink slots, applied by hand

THE VERIFIER NEVER CALLS THE HARNESS. Scoring a rollout through the repair loop would credit
the model for the repair node's work (reward leakage). The harness is used only to produce
rescue targets for classes no rollout could serve.
"""
