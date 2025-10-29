#!/usr/bin/env python3
"""
Command line tool template with subcommands: run, nsys, ncu, compare
"""

import argparse
import logging
import os, sys
import pandas as pd
import subprocess
import yaml
from typing import List, Optional

pd.options.display.float_format = '{:.3f}'.format

def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s'
    )

def load_defaults(config_name: str, verbose: bool = False) -> dict:
    #Checking if config_name is None
    config_name = config_name if config_name else 'runner.yaml'

    print(f"Opening {config_name}")
    with open(config_name, 'r') as file:
        yaml_doc = yaml.load(file, Loader=yaml.SafeLoader)

    if verbose:
        print(yaml.safe_dump(yaml_doc))

    return yaml_doc

def build_larnd_cmd(larnd_config: dict) -> str:
    """Bild larnd-sim command for ncu and nsys."""
    config = larnd_config['config']
    rng_seed = larnd_config['rng_seed']
    input_file = larnd_config['input_file']
    output_file = larnd_config['output_file']

    larnd_sim_cmd = f" simulate_pixels.py {config} --input_filename {input_file} --output_filename {output_file}"
    larnd_sim_cmd += f" --rand_seed {rng_seed}"

    if larnd_config.get('n_events', None):
        larnd_sim_cmd += f" --n_events {larnd_config['n_events']}"

    compression = larnd_config.get('compression', 'lzf')
    larnd_sim_cmd += f" --compression {compression}"

    return larnd_sim_cmd

def cmd_run(args: argparse.Namespace, config: str) -> int:
    """Handle the 'run' subcommand to run larnd-sim."""
    logger = logging.getLogger(__name__)
    larnd_config = config['larnd-sim']
    cmd = f"simulate_pixels.py {args.config}"

    config = args.config if args.config else larnd_config['config']
    logger.info(f"Running larnd-sim with config: {config}")

    input_file = args.input if args.input else larnd_config['input_file']
    logger.info(f"Input edep-sim hdf5: {input_file}")

    output_file = args.output if args.output else larnd_config['output_file']
    logger.info(f"Output filename: {output_file}")

    default_seed = args.rand_seed if args.rand_seed else larnd_config.get('rng_seed', 321)
    compression = larnd_config.get('compression', None)

    cmd = f"simulate_pixels.py {config} --input_filename {input_file} --output_filename {output_file}"
    cmd += f" --rand_seed {default_seed}"

    n_events = args.nevents if args.nevents else larnd_config.get('n_events', None)
    if n_events:
        logger.info(f"Running {n_events} events")
        cmd += f" --n_events {n_events}"

    if compression:
        logger.info(f"Compressing with {compression}")
        cmd += f" --compression {compression}"

    if args.args:
        logger.info(f"Adding the following arguments {args.args}")
        cmd += f" {args.args}"

    if args.dry_run:
        print(f"DRY RUN -- Complete command to run:\n {cmd}")
        return 0
    else:
        if args.force:
            if os.path.exists(output_file):
                print(f"Deleting existing {output_file}")
                os.remove(output_file)

        print(f"Complete command:\n {cmd}")
        ret = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        return ret.returncode

def cmd_nsys(args: argparse.Namespace, config: dict) -> int:
    """Handle the 'nsys' subcommand for Nsight Systems profiling."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running Nsight Systems profiling")
    larnd_config = config['larnd-sim']
    nsys_config = config['nsys']

    nsys = nsys_config['exec']
    cmd = f"{nsys} profile"
    cmd += " --cuda-memory-usage=true --python-backtrace=cuda --python-sampling=true"

    if args.force:
        cmd += " --force-overwrite=true"

    output_dir = nsys_config.get('output_dir', '.')
    output_file = args.output if args.output else nsys_config.get('output_file', None)
    if output_file:
        logger.info(f"Profile output: {output_file}")
        output_path = os.path.join(output_dir, output_file)
        cmd += f" -o {output_path}"

    if args.args:
        logger.info(f"Adding the following arguments {args.args}")
        cmd += f" {args.args}"

    cmd += build_larnd_cmd(larnd_config)
    if args.dry_run:
        print(f"DRY RUN -- complete command to run:\n {cmd}")
        return 0
    else:
        if args.force:
            lar_output = larnd_config['output_file']
            if os.path.exists(lar_output):
                print(f"Deleting existing {lar_output}")
                os.remove(lar_output)

        print(f"Complete command:\n {cmd}")
        ret = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        return ret.returncode

def cmd_ncu(args: argparse.Namespace, config: dict) -> int:
    """Handle the 'ncu' subcommand for Nsight Compute profiling."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running Nsight Compute profiling")
    larnd_config = config['larnd-sim']
    ncu_config = config['ncu']

    ncu = ncu_config['exec']
    cmd = f"{ncu} --nvtx"

    metrics = args.set if args.set else ncu_config.get('set', 'detailed')
    cmd += f" --set {metrics}"

    kernels = args.kernels if args.kernels else ncu_config['kernels']
    # Extract string from list, joining multiple kernels to form a regex pattern
    if len(kernels) > 1 and isinstance(kernels, list):
        kernels = "|".join(kernels)
    else:
        kernels = kernels[0]

    # Which invocation of the kernel to profile
    num_invoc = ncu_config.get('invocation', 5)
    cmd += f' --kernel-id "::regex:{kernels}:{num_invoc}"'

    if args.force:
        cmd += " --force-overwrite=true"

    output_dir = ncu_config.get('output_dir', '.')
    output_file = args.output if args.output else ncu_config.get('output_file', None)
    if output_file:
        logger.info(f"Profile output: {output_file}")
        output_path = os.path.join(output_dir, output_file)
        cmd += f" -o {output_path}"

    if args.args:
        logger.info(f"Adding the following arguments {args.args}")
        cmd += f" {args.args}"

    cmd += build_larnd_cmd(larnd_config)
    if args.dry_run:
        print(f"DRY RUN -- complete command to run:\n {cmd}")
        return 0
    else:
        print(f"Complete command:\n {cmd}")
        tmp = subprocess.run('dcgmi profile --pause', shell=True)
        ret = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        tmp = subprocess.run('dcgmi profile --resume', shell=True)
        return ret.returncode

def cmd_compare(args: argparse.Namespace, config: dict) -> int:
    """Handle the 'compare' subcommand for comparing larnd-sim output."""
    logger = logging.getLogger(__name__)
    compare_config = config['compare']

    ref_file = args.reference if args.reference else compare_config['reference']
    sim_file = args.file

    logger.info(f"Reference file: {ref_file}")
    logger.info(f"Target file: {sim_file}")
    cmd = f"diff_files.py --ref_file {ref_file} --sim_file {sim_file}"

    if args.strict:
        cmd += " --strict"

    if args.verbose:
        cmd += " --verbose"

    if args.dry_run:
        print(f"DRY RUN -- complete command to run:\n {cmd}")
        return 0
    else:
        print(f"Complete command:\n {cmd}")
        ret = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        return ret.returncode

def cmd_report(args: argparse.Namespace, config: dict) -> int:
    logger = logging.getLogger(__name__)
    nsys_config = config['nsys']
    report_config = config['report']

    nsys = nsys_config['exec']
    nsys_file = args.file
    nsys_report = args.report if args.report else report_config['report']
    nsys_format = report_config.get('format', 'csv')
    nsys_timeunit = report_config.get('timeunit', 'ms')
    nsys_dir = os.path.dirname(nsys_file)
    nsys_stats_file = os.path.splitext(nsys_file)[0] + f"_{nsys_report}.{nsys_format}"

    logger.info(f"Nsys report file: {nsys_file}")
    # logger.info(f"Output file: {args.output}")

    cmd = f"{nsys} stats --report {nsys_report} --format {nsys_format} --timeunit {nsys_timeunit}"
    if args.force:
        cmd += " --force-export=true --force-overwrite=true"
    cmd += f" --output . {nsys_file}"

    if args.dry_run:
        print(f"DRY RUN -- complete command to run:\n {cmd}")
        return 0
    else:
        print(f"Complete command:\n {cmd}")
        ret = subprocess.run(cmd, shell=True, capture_output=False, text=True)

    if nsys_report == "nvtx_sum":
        df = pd.read_csv(nsys_stats_file)
        rel_time = df.iloc[:, 1] / df.iloc[0, 1] * 100
        df.insert(loc=1, column='Rel. Time (%)', value=rel_time)
        df.drop(columns=['Time (%)', 'Style'], inplace=True)
        df = df.round(3)
        print(df)
        df.to_csv(os.path.splitext(nsys_stats_file)[0] + "_edit.csv", index=False)

    return 0

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Command line tool for running, profiling, and comparing larnd-sim"
    )

    # Global arguments
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '-y', '--yaml',
        help='YAML file containing common configuration options.'
    )
    parser.add_argument(
        '-d', '--dry_run',
        action='store_true',
        help='Print command but do not execute'
    )
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        default=False,
        help='Overwrite existing output files.'
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )

    # 'run' subcommand
    parser_run = subparsers.add_parser(
        'run',
        help='Run larnd-sim',
        description='Produce new larnd-sim output'
    )
    parser_run.add_argument(
        '-c', '--config',
        help='Larnd-sim configuration to use'
    )
    parser_run.add_argument(
        '-i', '--input',
        help='Input edep-sim hdf5 file'
    )
    parser_run.add_argument(
        '-o', '--output',
        help='Output larnd-sim file name'
    )
    parser_run.add_argument(
        '-n', '--nevents',
        type=int,
        help='Number of events to process'
    )
    parser_run.add_argument(
        '-s', '--rand_seed',
        type=int,
        help='Random number seed'
    )
    parser_run.add_argument(
        '--args',
        nargs=argparse.REMAINDER,
        help='Additional arguments to pass to target program'
    )

    # 'nsys' subcommand
    parser_nsys = subparsers.add_parser(
        'nsys',
        help='Run Nsight Systems profiling',
        description='Profile larnd-sim using NVIDIA Nsight Systems'
    )
    parser_nsys.add_argument(
        '-o', '--output',
        help='Output file for profile data'
    )
    parser_nsys.add_argument(
        '--args',
        nargs=argparse.REMAINDER,
        help='Additional arguments to pass to nsys'
    )

    # 'ncu' subcommand
    parser_ncu = subparsers.add_parser(
        'ncu',
        help='Run Nsight Compute profiling',
        description='Profile CUDA kernels using NVIDIA Nsight Compute'
    )
    parser_ncu.add_argument(
        '-o', '--output',
        help='Output file for profile data'
    )
    parser_ncu.add_argument(
        '--kernels',
        nargs='+',
        help='Specific kernels to profile'
    )
    parser_ncu.add_argument(
        '--set',
        choices=['basic', 'detailed', 'full'],
        help='Predefined metric set to collect'
    )
    parser_ncu.add_argument(
        '--args',
        nargs=argparse.REMAINDER,
        help='Additional arguments to pass to ncu'
    )

    # 'compare' subcommand
    parser_compare = subparsers.add_parser(
        'compare',
        help='Compare larnd-sim files',
        description='Compare two larnd-sim output files'
    )
    parser_compare.add_argument(
        'file',
        help='Target file to validate'
    )
    parser_compare.add_argument(
        '-o', '--output',
        help='Output file for comparison results'
    )
    parser_compare.add_argument(
        '-r', '--reference',
        help='Reference larnd-sim file'
    )
    parser_compare.add_argument(
        '--strict',
        action='store_true',
        help='Enable strict comparisons between files'
    )

    # 'report' subcommand
    parser_report = subparsers.add_parser(
        'report',
        help='Generate nsys profile summary',
        description='Generate summary of larnd-sim performance from nsys'
    )
    parser_report.add_argument(
        'file',
        help='Input nsys report file'
    )
    parser_report.add_argument(
        '-r', '--report',
        help='Nsys report to generate'
    )
    parser_report.add_argument(
        '-o', '--output',
        help='Output file for summary'
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    config_opts = load_defaults(args.yaml, args.verbose)

    # Dispatch to appropriate subcommand handler
    if args.command == 'run':
        return cmd_run(args, config_opts)
    elif args.command == 'nsys':
        return cmd_nsys(args, config_opts)
    elif args.command == 'ncu':
        return cmd_ncu(args, config_opts)
    elif args.command == 'compare':
        return cmd_compare(args, config_opts)
    elif args.command == 'report':
        return cmd_report(args, config_opts)
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())
