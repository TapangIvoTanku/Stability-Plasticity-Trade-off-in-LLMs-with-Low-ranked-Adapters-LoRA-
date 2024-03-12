
#datasets download/loading
from datasets import load_dataset

task_A = load_dataset("stacked-summaries/onlystacked-xsum-1024", "default",split="train")
task_B = load_dataset("shefali2023/webmd-data", split="train")
# print(f"Task_A: {len(task_A)}, Task_B: {len(task_B)}")

#preprocessing and prep
task_A_processed = task_A.map(lambda sample: {"input": sample['document'], "output": sample['summary']}, remove_columns = task_A.column_names)

def process_taskB(sample):
  output = {}
  input_output_lst = sample['Prompt'].replace("<s>[INST]", "").replace("</s>", "").split("[/INST]")
  output['input'] = input_output_lst[0]
  output['output'] = input_output_lst[-1]
  return output
task_B_processed = task_B.map(process_taskB, remove_columns=task_B.column_names)

temp_task_A = [i for i in task_A_processed]
temp_task_B = [i for i in task_B_processed]
# len(temp_task_A), len(temp_task_B)

import json

def save_lst_to_json(lst, output_dir):
    with open(output_dir, 'w') as file:
        file.write("\n".join([json.dumps(sample) for sample in lst]))

save_lst_to_json(temp_task_A[:-100], "/content/task_A_train_ds.jsonl")
save_lst_to_json(temp_task_A[-100:], "/content/task_A_test_ds.jsonl")

save_lst_to_json(temp_task_B[:-100], "/content/task_B_train_ds.jsonl")
save_lst_to_json(temp_task_B[-100:], "/content/task_B_test_ds.jsonl")

