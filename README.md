# LLM-Powered Multi-Agent Simulation for Climate Policy Evaluation

![Multi-Agent-Architecture](https://github.com/ai-r-ia/multi-agent-climate-policies/blob/main/multi_agent_arch.jpeg)

## Overview: A Novel Policy Testbed

This repository contains the code and results for a novel **Large Language Model (LLM)-based Multi-Agent Simulation (MAS)** framework designed to evaluate climate policies.

Traditional models (IAMs) often fail to capture **heterogeneity** and **bounded rationality**. Our framework addresses this by instantiating LLMs (e.g., Falcon-7B, Llama-8B) as diverse, empirically-calibrated agents (households and firms) with human-like reasoning and adaptive behavior.

The core experiment simulates the iterative refinement of the **EU's Carbon Border Adjustment Mechanism (CBAM)** policy, where AI agents provide feedback and a central policymaker agent updates the policy accordingly.

This system serves as a dynamic testbed to analyze **distributional effects**, **emergent dynamics**, and the **social robustness** of policies, providing micro-founded insights for cost-effective and socially feasible climate interventions.

---

## Key Simulation Structure

The simulation centers around a robust iterative loop:

1.  **Initial Policy**: The EU's CBAM description.
2.  **Agent Personas (6 total)**:
    * **Households (3)**: Middle-income climate-conscious, low-income cost-sensitive, high-income tech-savvy.
    * **Firms (3)**: Small manufacturing (cost-focused), large chemical (clean tech investing), medium renewable energy startup.
3.  **Iterative Process**:
    * Agents provide feedback on the current policy version.
    * The Policy Maker (simulated via GPT-5 in the research context) analyzes feedback, checks against **ethical guardrails**, and updates the policy.
    * The process repeats for a specified number of iterations.

---

## Research Documentation & Key Findings

### Research Paper
The full research paper documenting the CBAM simulation methodology, LLM agent architecture, and detailed findings is available here:

* [**Paper**](https://github.com/ai-r-ia/multi-agent-climate-policies/blob/main/paper.pdf)

### Key Simulation Findings
The results from the experiments demonstrate:
* **Policy Evolution:** Tracking how policy language is refined through iterative feedback cycles (e.g., moving from narrow to balanced proposals).
* **Model Performance:** Differences in response quality and policy judgment between LLM models (e.g., Falcon-7B excelled early; Llama performed worst overall).
* **Optimal Iteration:** Moderate policy feedback cycles yield **targeted, useful adjustments**, while excessive iteration can lead to generic outcomes.
* **Comparison:** Highlighting the differences in outcomes between "single" model mode and "mixed" model mode.
![scores](https://github.com/ai-r-ia/multi-agent-climate-policies/blob/main/scores.png)
---

## Files Description

### Simulation Scripts
* **`model_var.py`** - **Main simulation script.** This file executes the CBAM policy simulation, supports configurable modes (`single` or `mixed`), and manages the iterative policy refinement logic.

### Simulation Results
The JSON files below contain detailed data from various simulation runs.

#### 15-Iteration Runs (Model Comparison)
| File Name | Model | Mode | Iterations |
| :--- | :--- | :--- | :--- |
| `simulation_results_llama_single_15.json` | Llama-Guard-2-8B | Single | 15 |
| `simulation_results_falcon_single_15.json` | Falcon-7B-Instruct | Single | 15 |
| `simulation_results_falcon_mixed_15.json` | Falcon-7B-Instruct | Mixed | 15 |

#### Iteration Depth Runs (Falcon)
| File Name | Model | Iterations |
| :--- | :--- | :--- |
| `simulation_results_3.json` | Falcon-7B-Instruct | 3 |
| `simulation_results_5.json` | Falcon-7B-Instruct | 5 |
| `simulation_results_10.json` | Falcon-7B-Instruct | 10 |

### Data Format
Each result file follows the structure below, including agent responses and model metadata:

```json
{
  "model_used": "model_name",
  "simulation_mode": "single|mixed", 
  "iterations_completed": number,
  "iterations": [
    {
      "iteration": number,
      "policy_version": "current_policy_text",
      "agent_responses": [
        {
          "persona": "persona_description",
          "response": "full_agent_response",
          "response_summary": "summarized_response",
          "model_used": "model_name"
        }
      ]
    }
  ],
  "current_policy": "final_policy_version"
}
```
## Usage
To run new simulations:

Set Up: Configure your desired model choice, simulation mode (single or mixed), and iteration parameters in model_var.py.

Authentication: Set your Hugging Face token to grant access to the LLM models.

Run: Execute the model_var.py script.

Output: Results are automatically saved as JSON files in the appropriate directory.

Note: Response times vary significantly based on hardware and the complexity of the LLM model being used. Some response fields may be empty due to model filtering or generation issues.
