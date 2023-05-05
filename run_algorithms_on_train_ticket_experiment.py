import json
import shlex
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO
import click
import pandas as pd
from loguru import logger

from run_algorithm import executor_derived


def get_average_latency_data_command(
        service: str, experiment_datetime: datetime, input_dir: Path, data_output_dir: Path
):
    return (
        f"python run_prepare_input.py -a \"{experiment_datetime.strftime('%Y-%m-%d %H:%M')}\" "
        f"--output-metric average_client_cost "
        f"--derived total_client_cost,divide,count "
        f"-t timestamp "
        f"-m count,cost,proc,succ,client_cost,stall_rate "
        f"--column-names None "
        f"--extra-eval-columns \"total_client_cost=client_cost*count\" "
        f"--read-pickle "
        f"--timestamp-column-type timestamp "
        f"--query \"serviceName==\'{service}\'\" "
        f"--drop-columns operationName,url,kind,serviceName "
        f"--window-length 5 "
        f"--output-dir {str(data_output_dir)} "
        f"{str(input_dir / 'metrics' / 'squeeze_metrics.pkl')} "
    )


def get_stall_rate_data_command(
        service: str, experiment_datetime: datetime, input_dir: Path, data_output_dir: Path
):
    return (
        f"python run_prepare_input.py -a \"{experiment_datetime.strftime('%Y-%m-%d %H:%M')}\" "
        f"--output-metric total_stall_rate "
        f"--derived total_stall,divide,count "
        f"-t timestamp "
        f"-m count,cost,proc,succ,client_cost,stall_rate "
        f"--column-names None "
        f"--extra-eval-columns \"total_stall=stall_rate*count\" "
        f"--read-pickle "
        f"--timestamp-column-type timestamp "
        f"--query \"serviceName==\'{service}\'\" "
        f"--drop-columns operationName,url,kind,serviceName "
        f"--output-dir {str(data_output_dir)} "
        f"{str(input_dir / 'metrics' / 'squeeze_metrics.pkl')} "
    )


@click.command()
@click.option("--input-path", "-i", type=str)
def main(input_path: str):
    input_dir = Path(input_path)
    assert input_dir.is_dir(), input_dir
    del input_path

    experiment_type, experiment_datetime_string = input_dir.name.split("-at-")
    experiment_datetime = datetime.strptime(experiment_datetime_string, "%Y-%m-%d-%H-%M")
    experiment_datetime += timedelta(minutes=1)

    data_output_dir = Path("data_train_ticket")
    data_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"{experiment_type=}")
    ground_truth_info_string_io = StringIO()
    if experiment_type in {
        "istio-basic-get_station-delay", "istio-basic-get-station-delay",
        "istio-basic-post_travel-delay", "pod-network-delay",
        "istio-train-train_type-delay",
        "istio-price-train_type-delay",
        "istio-food-end_station-delay",
        "istio-food-start_station-delay",
    }:
        if experiment_type.startswith("istio"):
            with open(input_dir / "fault.yml", "r") as f:
                print(f"faults.yml: \n{f.read()}", file=ground_truth_info_string_io)
            if experiment_type.startswith("istio-basic"):
                target_service = "ts-basic-service"
            elif experiment_type.startswith("istio-train"):
                target_service = "ts-train-service"
            elif experiment_type.startswith("istio-price"):
                target_service = "ts-price-service"
            elif experiment_type.startswith("istio-food"):
                target_service = "ts-food-service"
            else:
                raise NotImplementedError(f"{experiment_type=}")
        elif experiment_type == "pod-network-delay":
            with open(input_dir / "chaos" / "ground_truths.json", "r") as f:
                _ground_truths = json.load(f)
                print(f"{_ground_truths=}", file=ground_truth_info_string_io)
                target_service = '-'.join(_ground_truths[0].split("-")[:-1])
        else:
            raise NotImplementedError

        command = get_average_latency_data_command(target_service, experiment_datetime, input_dir, data_output_dir)
        logger.info(f"{command=}")
        subprocess.run(shlex.split(command))
        command = get_stall_rate_data_command(target_service, experiment_datetime, input_dir, data_output_dir)
        logger.info(f"{command=}")
        subprocess.run(shlex.split(command))
        input_file_basename = f"{int(experiment_datetime.timestamp())}.squeeze_metrics.pkl.average_client_cost"
        results = []

        def _run_algorithm(__algorithm):
            _rets = []
            for _metric_name in ["average_client_cost", "total_stall_rate"]:
                _ret = executor_derived(
                    [
                        data_output_dir / f"{int(experiment_datetime.timestamp())}.squeeze_metrics.pkl.{_metric_name}.a.csv",
                        data_output_dir / f"{int(experiment_datetime.timestamp())}.squeeze_metrics.pkl.{_metric_name}.b.csv",
                        ],
                    output_path=Path("./output"), injection_info=pd.DataFrame(), algorithm=__algorithm,
                )
                _ret["metric_name"] = _metric_name
                _rets.append(_ret)
            return _rets
    else:
        raise NotImplementedError
    for algorithm in ["PSQ", "SQ", "ADT", "RAD", "APR", "HS", "MID", "IAP"]:
    # for algorithm in ["APR"]:  # DEBUG
        logger.info(f"Running {algorithm}")
        __rets = _run_algorithm(algorithm)
        for __ret in __rets:
            results.append({
                "algorithm": algorithm,
                "failure": input_dir.name,
                "elapsed_time": __ret.get("elapsed_time", float("NaN")),
                "root_cause": __ret.get("root_cause", ""),
                "metric_name": __ret.get("metric_name", ""),
                "n_rc": len(__ret.get("root_cause", "").split(";")),
            })
    ret_df = pd.DataFrame.from_records(results)
    with open(f"output/train_ticket_logs/experiments.log", "a+") as f:
        logger.add(f)
        logger.info(f"\n{input_dir}")
        logger.info(f"\n{ret_df.to_csv(index=False)}")
        logger.info(ground_truth_info_string_io.getvalue())


if __name__ == '__main__':
    main()
