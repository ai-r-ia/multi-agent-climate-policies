# Results Folder

This folder contains the results and analysis from CBAM (Carbon Border Adjustment Mechanism) policy simulation experiments using large language models.

## Overview

The experiments simulate a policy-making process where AI agents representing different stakeholders (households and firms) provide feedback on the EU's Carbon Border Adjustment Mechanism policy. The policy maker then iteratively refines the policy based on this feedback.

## Files Description

### Simulation Scripts

- **`model_var.py`** - Main simulation script that runs the CBAM policy simulation
  - Supports both Falcon-7B and Llama-Guard-2-8B models
  - Configurable simulation modes: "single" (one model) or "mixed" (random model selection)
  - Simulates 6 different agent personas (3 households, 3 firms)
  - Implements iterative policy refinement based on agent feedback


### Simulation Results

The JSON files contain detailed results from different simulation runs:

#### Model-Specific Results (15 iterations)
- **`simulation_results_llama_single_15.json`** - Llama model, single mode, 15 iterations
- **`simulation_results_falcon_single_15.json`** - Falcon model, single mode, 15 iterations  
- **`simulation_results_falcon_mixed_15.json`** - Falcon model, mixed mode, 15 iterations

#### Iteration-Specific Results (Falcon model)
- **`simulation_results_3.json`** - Falcon model, 3 iterations
- **`simulation_results_5.json`** - Falcon model, 5 iterations
- **`simulation_results_10.json`** - Falcon model, 10 iterations

### Research Documentation

- **`paper.pdf`** - Research paper documenting the CBAM simulation methodology and findings

## Simulation Structure

Each simulation run includes:

1. **Initial Policy**: EU's Carbon Border Adjustment Mechanism description
2. **Agent Personas**:
   - **Households**: middle-income climate-conscious, low-income cost-sensitive, high-income tech-savvy
   - **Firms**: small manufacturing (cost-focused), large chemical (clean tech investing), medium renewable energy startup
3. **Iterative Process**: 
   - Agents provide feedback on current policy
   - Policy maker analyzes feedback and updates policy
   - Process repeats for specified number of iterations

## Data Format

Each result file contains:
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
          "agent_type": "household|firm",
          "persona": "persona_description",
          "response": "full_agent_response",
          "response_summary": "summarized_response",
          "response_time_sec": number,
          "model_used": "model_name"
        }
      ]
    }
  ],
  "current_policy": "final_policy_version"
}
```

## Key Findings

The simulations demonstrate:
- How different AI models respond to policy feedback scenarios
- Evolution of policy language through iterative refinement
- Response time variations between models and personas
- Differences between single-model vs mixed-model approaches

## Usage

To run new simulations:
1. Configure model choice and simulation parameters in `model_var.py`
2. Set your Hugging Face token for model access
3. Run the script to generate new results
4. Results are automatically saved as JSON files


## Notes

- Some response fields may be empty due to model filtering or generation issues
- Response times vary significantly based on hardware and model complexity
- The simulations provide insights into AI model behavior in policy-making contexts
