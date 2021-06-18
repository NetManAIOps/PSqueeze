from pathlib import Path
from pprint import pprint

import click
import pandas as pd

from run_algorithm import executor, executor_derived
from ImpAPTr import ImpAPTr
from MID import MID
from post_process import post_process
from squeeze import Squeeze, SqueezeOption
from utility import AC


@click.command('Runner')
@click.option(
    "--algorithm", default="psqueeze", help="algorithm name",
)
@click.option("--derived", is_flag=True, help="means we should read {timestamp}.a.csv and {timestamp}.b.csv")
@click.option("--toint", is_flag=True, help="round measure values to integer")
@click.option("--n-ele", default=1, help="for ImpAPTr only")
@click.option("--output-dir", "-o", help="save outputs", default="./output")
@click.argument("input_file_list", nargs=-1)
def main(input_file_list, output_dir, **kwargs):
    input_file_list = list(map(Path, input_file_list))
    output_dir = Path(output_dir)
    derived: bool = kwargs['derived']
    if derived:
        result = executor_derived(
            input_file_list, output_path=output_dir, injection_info=pd.DataFrame(), **kwargs
        )
    else:
        result = executor(
            input_file_list[0], output_path=output_dir, injection_info=pd.DataFrame(), **kwargs
        )
    if kwargs['algorithm'].lower() in {'psq', 'psqueeze'}:
        post_process([result])
    pprint(result)


if __name__ == '__main__':
    main()
