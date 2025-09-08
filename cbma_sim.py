import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

hf_token = "hugging face token, sorry couldn't post it online :)"
os.environ["HF_TOKEN"] = hf_token

use_tpu = False
try:
    import torch_xla.core.xla_model as xm

    tpu_device = xm.xla_device()
    use_tpu = True
    print("TPU detected. Using TPU.")
except ImportError:
    print("TPU not available. Falling back...")
if not use_tpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU detected. Using GPU.")
    else:
        device = torch.device("cpu")
        print("Using CPU (slow).")
else:
    device = tpu_device

CANDIDATE_MODELS = [
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-3b-instruct",
    "tiiuae/falcon-1b-instruct",
]
model, tokenizer, MODEL_NAME = None, None, None
for model_name in CANDIDATE_MODELS:
    try:
        print(f"Trying to load: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_auth_token=os.environ["HF_TOKEN"]
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=os.environ["HF_TOKEN"],
            torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
        ).to(device)
        model.eval()
        print(f"Successfully loaded: {model_name}")
        MODEL_NAME = model_name
        break
    except RuntimeError as e:
        print(f"Failed to load {model_name}: {e}")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Unexpected error with {model_name}: {e}")
if model is None:
    raise RuntimeError(
        "All candidate models failed to load. Try a smaller model manually."
    )

policy_base = (
    "The European Union has introduced the Carbon Border Adjustment Mechanism (CBAM), "
    "imposing a carbon price on imports to encourage cleaner production globally, "
    "aimed at preventing carbon leakage and supporting the EU's climate targets."
)
households = [
    "middle-income, climate-conscious household",
    "low-income, cost-sensitive household",
    "high-income, tech-savvy household",
]
firms = [
    "small manufacturing firm focused on costs",
    "large chemical company investing in clean tech",
    "medium-sized renewable energy startup",
]


def summarize_agent_response(full_text):
    # Only the pure agent feedback text, no agent prompt instruction
    prompt = (
        "You are an assistant summarizing community feedback.\n"
        "Summarize the key concerns, suggestions, and overall sentiment "
        "in this text in 2-3 concise sentences and return output with only the summarized text:\n\n"
        f"{full_text}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.65,
        top_p=0.9,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # print("DEBUG summary: ", summary)
    # Remove any repeated prompt from output if echoed
    if summary.startswith(prompt):
        summary = summary[len(prompt) :].strip()
    return summary


class PolicyMaker:
    def __init__(self, initial_policy, tokenizer, model, device, use_tpu=False):
        self.current_policy = initial_policy
        self.history = [initial_policy]
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.use_tpu = use_tpu
        self.latest_summary = ""

    def _analyze_feedbacks(self, agent_feedback_summaries):
        feedback_text = "\n".join(agent_feedback_summaries)
        prompt = (
            f"Here are summarized citizens' and firms' feedbacks on a climate policy:\n"
            f"{feedback_text}\n\n"
            "Summarize the overall sentiment (positive/negative/mixed) "
            "and list the top 2 concerns or requests in a short JSON format like:\n"
            '{"sentiment": "mixed", "concerns": ["high costs", "need more innovation support"]}'
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
        )
        if self.use_tpu:
            import torch_xla.core.xla_model as xm

            xm.mark_step()
        self.latest_summary = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()
        return self.latest_summary

    def update_policy(self, agent_feedback_summaries):
        analysis = self._analyze_feedbacks(agent_feedback_summaries)

        # Create brief summary if analysis is JSON
        try:
            import json

            analysis_json = json.loads(analysis)
            sentiment = analysis_json.get("sentiment", "mixed")
            concerns = analysis_json.get("concerns", [])
            brief_analysis = (
                f"Sentiment: {sentiment}. Main concerns: {', '.join(concerns)}."
            )
        except Exception:
            brief_analysis = "Summary of stakeholder feedback."

        prompt = (
            f"Current policy:\n{self.current_policy}\n\n"
            f"Based on this feedback summary: {brief_analysis}\n\n"
            "Write a concise revised version of the policy in 2-3 sentences.\n"
            "Respond ONLY with the updated policy text without any explanations.\n"
            "### Response:\n"
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        if self.use_tpu:
            import torch_xla.core.xla_model as xm

            xm.mark_step()

        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Extract only after ### Response:
        if "### Response:" in raw_output:
            new_policy = raw_output.split("### Response:")[1].strip()
        else:
            new_policy = raw_output

        self.current_policy = new_policy
        self.history.append(new_policy)
        return new_policy


def ask_agent(persona, policy):
    prompt = (
        f"You are a {persona}.\n"
        f"Policy: {policy}\n"
        f"Explain in no more than 150 words your feelings, reactions (positive or negative), "
        f"and intended actions. You can suggest changes to the policy."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        device
    )
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=3,
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if full_text.startswith(prompt):
        response = full_text[len(prompt) :].strip()
    else:
        response = full_text
    if use_tpu:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    return response


def run_simulation():
    policymaker = PolicyMaker(policy_base, tokenizer, model, device, use_tpu)
    all_logs = []
    num_iterations = 5
    save_path = "/kaggle/working/simulation_results.json"

    for iteration in range(num_iterations):
        print(f"\n--- Policy Iteration {iteration + 1} ---")
        current_policy = policymaker.current_policy
        print(f"Current Policy:\n{current_policy}\n")

        full_responses = []
        summarized_responses = []
        iteration_log = {
            "iteration": iteration + 1,
            "policy_version": current_policy,
            "agent_responses": [],
        }

        for persona in households + firms:
            start_time = time.time()
            full_response = ask_agent(persona, current_policy)
            summary = summarize_agent_response(full_response)
            elapsed = time.time() - start_time

            print(f"Agent: {persona} Response time: {elapsed:.2f}s")
            # print full response raw (with emojis if present)
            print(full_response, "\n")
            # print summarized version
            print("Summary:\n", summary, "\n")

            iteration_log["agent_responses"].append(
                {
                    "agent_type": "household" if persona in households else "firm",
                    "persona": persona,
                    "response": full_response,
                    "response_summary": summary,
                    "response_time_sec": elapsed,
                }
            )

            full_responses.append(full_response)
            summarized_responses.append(summary)

        new_policy = policymaker.update_policy(summarized_responses)
        print(f"Policy updated:\n{new_policy}\n")

        all_logs.append(iteration_log)

        with open(save_path, "w") as f:
            json.dump(
                {
                    "model_used": MODEL_NAME,
                    "iterations_completed": iteration + 1,
                    "iterations": all_logs,
                    "current_policy": policymaker.current_policy,
                },
                f,
                indent=4,
            )

        print(f"Progress saved after iteration {iteration + 1}")

    print(f"Simulation complete. Results saved to {save_path}")


if __name__ == "__main__":
    run_simulation()
