from pathlib import Path
from pprint import pprint

import click
import pandas as pd
from loguru import logger
import sys
from post_process import post_process
from run_algorithm import executor, executor_derived


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
    logger.remove()
    logger.add(
        sys.stdout, level="DEBUG",
        format="[<green>{time}</green>, <blue>{level}</blue>] <white>{message}</white>"
    )
    input_file_list = list(map(Path, input_file_list))
    output_dir = Path(output_dir)
    derived: bool = kwargs.pop('derived')
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
