# PSqueeze
Implementation and dataset for *Generic and Robust Root Cause Localization for Multi-Dimensional Data in Online Service Systems* (Accepted by Journal of Systems of Software), which extends our previous [conference version](https://github.com/netmanaiops/squeeze)

Preprint: https://arxiv.org/abs/2305.03331

## Requirements
At least `python>=3.6` is required.
``` bash
pip install -r requirements.txt
```

A virtual environment is strongly recommended.

## Datasets
All datasets are available at Tsinghua Cloud: https://cloud.tsinghua.edu.cn/d/aa4102a5d1614e57bc36/

For simulation datasets, the ground-truth root causes are in `injection_info.csv` in each subfolder.

For injection datasets, each subdirectory contains monitoring data for one fault injection. Their ground-truth root causes are indicated by the subdirectory names. 

## Usage
### Simulation Datasets
For convenience, `run_exp.sh` provide a script to run all experiments.

Alternatively, you can run each experiment by yourself.
```
python run_algorithm.py --help
```
```
Usage: run_algorithm.py [OPTIONS]

  :param name: :param input_path: :param output_path: :param num_workers:
  :param kwargs: :return:

Options:
  --name TEXT            name of this setting
  --input-path TEXT      will read data from {input_path}/{name}
  --output-path TEXT     if {output_path} is a dir, save to
                         {output_path}/{name}.json; otherwise save to
                         {output_path}
  --num-workers INTEGER  num of processes
  --derived              means we should read {timestamp}.a.csv and
                         {timestamp}.b.csv
  --help                 Show this message and exit.
```

``` 
python run_evaluation.py --help
```
```
Usage: run_evaluation.py [OPTIONS]

Options:
  -i, --injection-info TEXT  injection_info.csv file
  -p, --predict TEXT         output json file
  -c, --config TEXT          config json file
  -o, --output-path TEXT     output path
  --help                     Show this message and exit.
```

The config json file should contain the attribute names, e.g.:

```
{
  "columns": [
    "a", "b", "c", "d"
  ]
}
```

### Injection Datasets
Run this command: `python3 run_algorithms_on_train_ticket_experiment.py [path_to_injection_data]`
For example, `python3 run_algorithms_on_train_ticket_experiment.py istio-food-end_station-delay-at-2022-10-09-23-00`
