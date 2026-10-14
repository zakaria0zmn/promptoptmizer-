"""
DSPy Optimization Pipeline with Local Ollama
--------------------------------------------
Target Version: DSPy v2.5+ / v2.6+
For legacy DSPy (< v2.5), see comments in Section 2 for dspy.OllamaLocal syntax.
"""

import re

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

# ==============================================================================
# SECTION 1: Configuration & Local LM Setup
# ==============================================================================
# Select a local model tag.
OLLAMA_MODEL_NAME = "qwen2.5-7b:latest"
# Alternative: OLLAMA_MODEL_NAME = "qwen3.8-9b:latest"
OLLAMA_API_BASE = "http://localhost:11434/v1"

print(f"[*] Initializing DSPy LM with Ollama model: {OLLAMA_MODEL_NAME}...")

# DSPy v2.5+ uses a unified dspy.LM abstraction backed by LiteLLM.
# Ollama's OpenAI-compatible API is exposed under /v1.
lm = dspy.LM(
    model=f"ollama_chat/{OLLAMA_MODEL_NAME}",
    api_base=OLLAMA_API_BASE,
    api_key="ollama",  # Dummy key satisfies DSPy's credential check; Ollama ignores it
    temperature=0.0,  # Zero temperature for deterministic evaluation
    max_tokens=600,
)
dspy.configure(lm=lm)

# NOTE (Legacy DSPy < v2.5 fallback):
# If using an older version where dspy.LM is unavailable, use:
# lm = dspy.OllamaLocal(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_API_BASE, temperature=0.0)
# dspy.settings.configure(lm=lm)


# ==============================================================================
# SECTION 2: Task Definition (Signature & Module)
# ==============================================================================
# SWAP PLACEHOLDER: Replace this Signature with your custom inputs and outputs.
class SupportTicketTriage(dspy.Signature):
    """Classify technical support ticket severity and extract the core root issue."""
    ticket_text: str = dspy.InputField(desc="Raw support request or bug report")
    severity: str = dspy.OutputField(desc="Must be exactly one of: 'Low', 'Medium', 'High', 'Critical'")
    issue_summary: str = dspy.OutputField(desc="A concise summary of the issue under 10 words")


# Standard pipeline module using ChainOfThought to encourage intermediate reasoning
class SupportTriageModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # ChainOfThought generates a rationale before producing the output fields
        self.classifier = dspy.ChainOfThought(SupportTicketTriage)

    def forward(self, ticket_text: str):
        return self.classifier(ticket_text=ticket_text)


# ==============================================================================
# SECTION 3: Dataset (Training & Validation Sets)
# ==============================================================================
# Format: Use dspy.Example with all fields, then designate inputs via .with_inputs()
raw_train_data = [
    {
        "ticket_text": "Production database cluster primary node is failing disk health checks and read replicas are out of sync.",
        "severity": "Critical",
        "issue_summary": "Database cluster failure and replica desync",
    },
    {
        "ticket_text": "Our payment gateway is returning 500 errors on checkout for all EU customers.",
        "severity": "Critical",
        "issue_summary": "Payment checkout failing for EU users",
    },
    {
        "ticket_text": "Typo on the terms of service page under section 4.2 heading.",
        "severity": "Low",
        "issue_summary": "Typo in terms of service",
    },
    {
        "ticket_text": "Users report that dark mode toggle occasionally resets after refreshing the browser.",
        "severity": "Low",
        "issue_summary": "Dark mode setting resets on refresh",
    },
    {
        "ticket_text": "PDF invoice export takes 45 seconds to generate instead of the usual 3 seconds.",
        "severity": "Medium",
        "issue_summary": "Slow invoice generation latency",
    },
    {
        "ticket_text": "The CSV customer export job failed last night with memory allocation error.",
        "severity": "Medium",
        "issue_summary": "Nightly CSV export failed on memory",
    },
    {
        "ticket_text": "API authentication endpoint is throwing 401 errors for valid tokens intermittently.",
        "severity": "High",
        "issue_summary": "Intermittent auth failures for valid tokens",
    },
    {
        "ticket_text": "Webhook delivery system is dropping payloads when incoming traffic spikes above 200 req/sec.",
        "severity": "High",
        "issue_summary": "Webhook queue drops events under load",
    },
]

raw_dev_data = [
    {
        "ticket_text": "Total outage: the landing page and authentication server are unreachable globally.",
        "severity": "Critical",
        "issue_summary": "Total service outage across global nodes",
    },
    {
        "ticket_text": "Can we change the profile avatar border color from blue to slate grey?",
        "severity": "Low",
        "issue_summary": "UI cosmetic color adjustment request",
    },
    {
        "ticket_text": "User search queries take over 12 seconds when filtering by date range.",
        "severity": "Medium",
        "issue_summary": "High latency in filtered search queries",
    },
    {
        "ticket_text": "Single Sign-On SAML integration is failing for our largest enterprise client.",
        "severity": "High",
        "issue_summary": "Enterprise SSO SAML integration broken",
    },
]

# Transform into dspy.Example objects
trainset = [dspy.Example(**item).with_inputs("ticket_text") for item in raw_train_data]
devset = [dspy.Example(**item).with_inputs("ticket_text") for item in raw_dev_data]

"""
HOW TO LOAD YOUR OWN DATA:
--------------------------
import json
with open("my_data.json", "r") as f:
    raw = json.load(f)
trainset = [dspy.Example(**row).with_inputs("ticket_text") for row in raw["train"]]
devset = [dspy.Example(**row).with_inputs("ticket_text") for row in raw["dev"]]
"""


# ==============================================================================
# SECTION 4: Custom Evaluation Metric
# ==============================================================================
def triage_evaluation_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Evaluates both severity exact match and reasonable summary generation.
    Returns a score between 0.0 and 1.0.
    
    - trace is passed by DSPy optimizers to inspect intermediate execution steps;
      leave default check logic intact when trace is None or provided.
    """
    if not hasattr(prediction, "severity") or not prediction.severity:
        return 0.0

    # Clean strings: strip whitespace and punctuation
    gold_severity = example.severity.strip().lower()
    pred_severity = re.sub(r"[^\w]", "", prediction.severity.strip().lower())

    # Exact match on categorical severity (weight: 0.7)
    severity_match = 1.0 if gold_severity == pred_severity else 0.0

    # Sanity check on summary length and non-empty status (weight: 0.3)
    summary_valid = 0.0
    if hasattr(prediction, "issue_summary") and prediction.issue_summary:
        words = prediction.issue_summary.strip().split()
        if 2 <= len(words) <= 15:
            summary_valid = 1.0

    # Composite score
    score = (0.7 * severity_match) + (0.3 * summary_valid)
    return score


# ==============================================================================
# SECTION 5: Baseline Evaluation
# ==============================================================================
print("\n" + "=" * 60)
print("[*] Running Baseline Evaluation (Zero-Shot / Uncompiled)...")
print("=" * 60)

# num_threads=1 is critical for local Ollama to avoid queuing bottlenecks or GPU memory contention
evaluator = Evaluate(
    devset=devset,
    metric=triage_evaluation_metric,
    num_threads=1,
    display_progress=True,
    display_table=False,
)

baseline_program = SupportTriageModule()
baseline_result = evaluator(baseline_program)
baseline_score = (
    baseline_result.score
    if hasattr(baseline_result, "score")
    else float(baseline_result)
)
print(f"\n[+] Baseline Score: {baseline_score:.2f}%")


# ==============================================================================
# SECTION 6: Optimization via BootstrapFewShot
# ==============================================================================
print("\n" + "=" * 60)
print("[*] Compiling Program with BootstrapFewShot...")
print("=" * 60)

# BootstrapFewShot runs the program against the trainset using the LM itself,
# retains traces where triage_evaluation_metric >= metric_threshold,
# and injects the highest-performing demonstrations directly into the prompt.
optimizer = BootstrapFewShot(
    metric=triage_evaluation_metric,
    metric_threshold=0.8,         # Only accept demonstrations scoring >= 80%
    max_bootstrapped_demos=2,     # 2-3 demonstrations keep local model prompts compact
    max_labeled_demos=2,
    max_rounds=1,
)

compiled_program = optimizer.compile(
    student=SupportTriageModule(),
    trainset=trainset,
)
print("[+] Program compilation complete.")


# ==============================================================================
# SECTION 7: Optimized Evaluation, Persistence, and Verification
# ==============================================================================
print("\n" + "=" * 60)
print("[*] Evaluating Compiled Program...")
print("=" * 60)

optimized_result = evaluator(compiled_program)
optimized_score = (
    optimized_result.score
    if hasattr(optimized_result, "score")
    else float(optimized_result)
)
print(f"\n[+] Optimized Score: {optimized_score:.2f}% (Delta: {optimized_score - baseline_score:+.2f}%)")

# Save the compiled prompt state (includes generated few-shot examples and prefixes)
SAVE_PATH = "optimized_support_triage.json"
print(f"\n[*] Saving optimized program state to '{SAVE_PATH}'...")
compiled_program.save(SAVE_PATH)

# Verify loading mechanism
print("[*] Verifying reload from disk...")
reloaded_program = SupportTriageModule()
reloaded_program.load(SAVE_PATH)

test_query = "Automated nightly backups are silently terminating with exit code 137 since the kernel update."
inference_result = reloaded_program(ticket_text=test_query)

print("\n--- Verification Inference on Test Sample ---")
print(f"Input Ticket:  {test_query}")
print(f"Predicted Severity: {inference_result.severity}")
print(f"Issue Summary:      {inference_result.issue_summary}")
if hasattr(inference_result, "rationale"):
    print(f"Model Rationale:    {inference_result.rationale}")
print("-" * 45)
