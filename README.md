# PSqueeze
Implementation and dataset for *Generic and Robust Root Cause Localization for Multi-Dimensional Data in Online Service Systems*, which extends our previous [conference version](https://github.com/netmanaiops/squeeze)

## Requirements
At least `python>=3.6` is required.
``` bash
pip install -r requirements.txt
```

A virtual environment is strongly recommended.

## Datasets

Datasets `A, B1, B2, B3, B4, D` are on [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/db8495e4b7624674924c/).
The ground truth root cause sets are in `injection_info.csv` in each subfolder.

## Usage
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