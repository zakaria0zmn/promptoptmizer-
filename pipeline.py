import sys
import subprocess
import dspy

# 1. Connect to Ollama
TARGET_MODEL = "qwen2.5-7b:latest"
lm = dspy.LM(
    model=f"openai/{TARGET_MODEL}",
    api_base="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0.3,
    max_tokens=800,
)
dspy.configure(lm=lm)


# 2. Define the Prompt Optimizer Signature
class PromptOptimizerSignature(dspy.Signature):
    """
    You are an expert prompt engineer. Take a rough or brief prompt idea
    and rewrite it into a high-performance, structured system prompt.
    Include:
    - Clear Persona and Objective
    - Concrete Constraints, Tools, and Edge Cases
    - Strict Output Format Rules
    - 1 brief input/output demonstration
    """
    rough_idea: str = dspy.InputField(desc="The user's raw prompt idea")
    target_environment: str = dspy.InputField(desc="Where the prompt will run (e.g., Goose agent, Ollama)")
    optimized_prompt: str = dspy.OutputField(desc="The complete, production-ready prompt")


optimizer = dspy.Predict(PromptOptimizerSignature)


def copy_to_clipboard(text: str):
    """Copies text to the Windows clipboard using native PowerShell."""
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", "Set-Clipboard", "-Value", "$input"],
        input=text,
        text=True,
        check=True
    )
    print("\n[✔] Copied optimized prompt to Windows clipboard!")


def send_to_goose(prompt: str):
    """Launches Goose CLI with the optimized prompt."""
    print(f"\n[*] Dispatching to Goose...")
    try:
        subprocess.run(["goose", "run", "--text", prompt], check=True)
    except FileNotFoundError:
        print("[!] 'goose' command not found in PATH. Ensure Goose CLI is installed.")
    except subprocess.CalledProcessError as e:
        print(f"[!] Goose exited with error code {e.returncode}")


def send_to_ollama(prompt: str, model: str = TARGET_MODEL):
    """Launches an interactive Ollama run with the prompt."""
    print(f"\n[*] Dispatching to Ollama ({model})...\n")
    try:
        subprocess.run(["ollama", "run", model, prompt], check=True)
    except FileNotFoundError:
        print("[!] 'ollama' command not found in PATH.")


# 3. Interactive Loop
def main():
    print("=" * 65)
    print(" Prompt Optimizer -> Dispatcher (Goose / Ollama / Clipboard)")
    print("=" * 65)

    while True:
        try:
            rough_idea = input("\nEnter rough idea (or 'q' to quit): ").strip()
            if rough_idea.lower() in ("q", "quit", "exit"):
                break
            if not rough_idea:
                continue

            print("\n[*] Optimizing prompt with Qwen 2.5...")
            result = optimizer(
                rough_idea=rough_idea,
                target_environment="developer agent / local LLM"
            )
            optimized = result.optimized_prompt.strip()

            print("\n" + "=" * 25 + " OPTIMIZED PROMPT " + "=" * 25)
            print(optimized)
            print("=" * 68)

            # Action menu
            print("\nWhere do you want to send this?")
            print("  [1] Send to Goose (`goose run --text ...`)")
            print("  [2] Send to Ollama (`ollama run ...`)")
            print("  [3] Copy to Clipboard (paste into active session)")
            print("  [4] Discard & try again")

            choice = input("\nSelect [1-4] (default: 3): ").strip() or "3"

            if choice == "1":
                send_to_goose(optimized)
            elif choice == "2":
                send_to_ollama(optimized)
            elif choice == "3":
                copy_to_clipboard(optimized)
            elif choice == "4":
                print("[*] Discarded.")
            else:
                print("[!] Invalid option.")

        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)


if __name__ == "__main__":
    main()