"""
This file is part of TA-Eval-Rep.
Copyright (C) 2022 University of Luxembourg
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 3 of the License.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import os

sys.path.append('../')
 

from logparser.TreeGen  import benchmark_settings
from logparser.TreeGen import LogParser
from evaluation.utils.common import datasets, common_args, unique_output_dir
from evaluation.utils.evaluator_main import evaluator, prepare_results
from evaluation.utils.postprocess import post_average
import itertools

datasets_2k = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    "Spark",
    "Thunderbird",
    "BGL",
    "HDFS",
]

datasets_full = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Hadoop",
    "HealthApp",
    "OpenStack",
    "HPC",
    "Mac",
    "OpenSSH",
    # "Spark",
    # "Thunderbird",
    # "BGL",
    "HDFS",
]


if __name__ == "__main__":
    args = common_args()
    data_type = "full" if args.full_data else "2k"
    input_dir = f"../../{data_type}_dataset/"
    output_dir = f"../../result/result_TreeGen_{data_type}"
    # prepare result_file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result_file = prepare_results(
        output_dir=output_dir,
        otc=args.oracle_template_correction,
        complex=args.complex,
        frequent=args.frequent
    )

    if args.full_data:
        datasets = datasets_full
    else:
        datasets = datasets_2k
    for dataset in datasets:
        setting = benchmark_settings[dataset]
        log_file = setting['log_file'].replace("_2k", f"_{data_type}")
        indir = os.path.join(input_dir, os.path.dirname(log_file))
        if os.path.exists(os.path.join(output_dir, f"{dataset}_{data_type}.log_structured.csv")):
            parser = None
            print("parsing result exist.")
        else:
            parser = LogParser
        # run evaluator for a dataset
        # GRIDSEARCH
        
        GA_vectors=[]
        dyn_pars={#"Q":[i*0.1 for i in range(1)],
                 "NGEN":[15],
                 "NPOP":[75]
                }
        pars=list(dyn_pars.keys())
        
        #product of all possible combinations of the parameters
        combinations = list(itertools.product(*dyn_pars.values()))

        default_vector={"NPOP":100, "NGEN":50, "SEED":0, "CXPB":0.5, "MUTPB":0.2, "Q":0.0}
        for combi in combinations:
            vector=default_vector.copy()
            for i,par in enumerate(pars):
                vector[par]=combi[i]
            
            GA_vectors.append(vector)
        
            
        for GA_vec in GA_vectors :
            evaluator(
                dataset=dataset,
                input_dir=input_dir,
                output_dir=output_dir,
                log_file=log_file,
                LogParser=parser,
                param_dict={
                    'log_format': setting['log_format'], 'indir': indir, 'outdir': output_dir,'GA' : GA_vec, 'rex': setting['regex']
                },
                otc=args.oracle_template_correction,
                complex=args.complex,
                frequent=args.frequent,
                result_file=result_file
            ) # it internally saves the results into a summary file
        
    metric_file = os.path.join(output_dir, result_file)
    post_average(metric_file, f"TreeGen_{data_type}_complex={args.complex}_frequent={args.frequent}", args.complex, args.frequent)