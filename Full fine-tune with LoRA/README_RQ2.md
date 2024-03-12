
# Exploring Plasticity-Stability Trade-offs in LLMs: RQ2 Focus

This repository presents a detailed examination of adaptation strategies in large language models (LLMs), emphasizing the trade-offs between plasticity (the model's ability to learn new tasks) and stability (the model's ability to retain knowledge on previously learned tasks). The investigation is guided by Research Question 2 (RQ2) and aims to discern the effects of Full fine-tuning and Low-Rank Adaptation (LoRA) on these dimensions.

## Research Question 2

**RQ2**: To what extent do Full fine-tuning and Low-Rank Adaptation (LoRA) demonstrate plasticity-stability trade-offs in LLMs, considering the impact on task performance, model stability, and computational efficiency when adapting to new tasks?

## Hypotheses for RQ2

- **H2a**: Full fine-tuning and Low-Rank Adaptation (LoRA) show a trade-off between plasticity and stability in LLMs when adapting to new tasks. One method may exhibit superior task performance but lower model stability or higher computational efficiency than the other.

## Experimental Procedure

1. **Task A (Full Fine-Tuning)**: The model undergoes a comprehensive fine-tuning process on Task A, serving as the initial adaptation phase.
2. **Task B (LoRA)**: Utilizing the model fine-tuned on Task A, we apply Low-Rank Adaptation (LoRA) to adapt to Task B, exploring the potential for maintaining Task A's performance while efficiently learning Task B.

## Directory Structure

- **Scripts**:
  - `data_prep.py`: Scripts for downloading and preparing datasets for Task A and Task B.
  - `full_finetune_task_B.py`: Script detailing the full fine-tuning process for Task A.
  - `lora_task_B.py`: Script applying LoRA to the model for Task B adaptation.
  - `requirements.txt`: A list of Python package dependencies required for the project.

## Installation

To set up your environment to run the experiments, first ensure Python is installed on your system. Then, install the necessary dependencies:

```
pip install -r scripts/requirements.txt
```

## Running the Experiments

For full fine-tuning on Task A:

```
python scripts/full_finetune_task_B.py
```

To apply LoRA for Task B adaptation:

```
python scripts/lora_task_B.py
```

## Contributing

Contributions that enhance the analysis or improve upon the implementation are welcome. Please fork this repository and submit a pull request with your changes.

## License

Ivo Tanku Tapang

## Contact

For further inquiries or to discuss the findings, please contact [Ivo Tanku Tapang](mailto:itankutapang@gmail.com).
