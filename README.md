# Memprompt


## Running a job

_Please note that you need to `export OPENAI_API_KEY="YOUR_OPENAI_KEY` before running the scripts._


### Streaming with memory


```sh
python memprompt/stream_with_memory.py --task_file ${FILE} --job_id ${JOB_ID} --getting_clarification_probability ${CLARIFICATION_PROB}
```

- Where:

    - `FILE` is the path to the task file
    - `JOB_ID` is the job ID
    - `CLARIFICATION_PROB` is the probability of getting clarification from the user

For example, to run a job with 10 samples on the linguistic variation prompts with a clarification probability of 0.5, run:

```sh
python memprompt/stream_infer.py --task_file memprompt/tasks/linguistic/tasks_10.jsonl \
                                 --job_id  linguistic \
                                 --getting_clarification_probability 0.5
```


* After the job finishes, it'll create a log file in `memprompt/logs` in the form: 
`memprompt/logs/${task_type}_num_tasks=${num_samples}_clarification_prob=${clarification_prob}_job_id=${job_id}_${timestamp}.jsonl`.
For the sample script, the file should be: `memprompt/logs/linguistic_num_tasks=10_clarification_prob=0.5_2021-11-08_21:44:00.jsonl`.

### Stream with growing prompt

- The usage is similar to the streaming with memory, but the prompt grows as the job proceeds.

```sh
python memprompt/stream_with_growing_prompt.py  --task_file memprompt/tasks/linguistic/tasks_10.jsonl\ 
                                                --job_id  linguistic \
                                                --getting_clarification_probability 0.5
```

## Creating new tasks

- New tasks can be created by running the following command:


```sh
python memprompt/task_handler.py n task
```

Where:
    
        - `n` is the number of tasks to create
        - `task` is the task type (hin, pun, or linguistic)

- For example, to create a file with 500 tasks of type `hin` (Hindi prompts), run:

```sh
python memprompt/task_handler.py 500 hin
```

## Processing logs

To list the progressive scores in a matrix form that can loaded into some Google sheets to generate charts:

```sh
python memprompt/utils.py --path memprompt/logs/ --pattern "task_type="
```

- Where:

    - `path` is the path to a single jsonl log file or a directory containing multiple files.
    - `pattern` matches the files within a path directory to process logs.
    - `out_path` is the output path to store the results.



## Task files

The directory structure is `tasks/<task_name>/tasks_<num_tasks>.jsonl`.
The current task files are:
```
tasks/
├── hin
│   ├── tasks_1000.jsonl
│   └── tasks_100.jsonl
├── linguistic
│   ├── tasks_1000.jsonl
│   ├── tasks_100.jsonl
│   └── tasks_10.jsonl
└── pun
    ├── tasks_1000.jsonl
    └── tasks_100.jsonl
```
