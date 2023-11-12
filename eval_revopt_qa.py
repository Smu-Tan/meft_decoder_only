import os
import sys
import json
import torch
import logging

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    AutoModelForCausalLM,
    default_data_collator,
    DataCollatorForTokenClassification,
    set_seed,
    Trainer
)

from utils.utils import ModelArguments, DataTrainingArguments, LMEval, EvalArguments
from lm_eval import evaluator

from models import OPTForCausalLM


#### preparations
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EvalArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        print('sss')
        model_args, data_args, eval_args, = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            #finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    config.use_cache = False
    config.adapter_bottleneck_dim = model_args.adapter_bottleneck_dim
    config.layernorm_in_adapter = model_args.layernorm_in_adapter
    config.num_rev_layers = model_args.num_rev_layers
    config.x1_factor = model_args.x1_factor
    config.x2_factor = model_args.x2_factor
    config.f_arch = model_args.f_arch
    config.freeze_irreversible_layers = model_args.freeze_irreversible_layers
    config.sum = model_args.sum
    config.sum_scale = model_args.sum_scale

    tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    
    config.use_cache = False

    if eval_args.eval_zeroshot:
        ### *** Evaluate ZS ***
        model = OPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                #torch_dtype=torch.float16,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        model = model.to("cuda")

        # Evaluation
        results = {}
        logger.info("*** Evaluate ZS ***")
        lm_eval_model = LMEval(model, tokenizer)
        zs_results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=[data_args.dataset_name],
                batch_size=128,
                no_cache=True,
            )    

        # only useful when computing inference memory
        if torch.cuda.is_available() and eval_args.compute_memory:
            peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
            print(
                "Memory utilization",
                peak_memory,
                "GB"
            )
            zs_results['Memory_utilization'] = '{} GB'.format(peak_memory)
        del zs_results['config']['model']

        ### save all results
        with open(eval_args.output_dir+'all_results.json', 'r') as file:
            all_results = json.load(file)
        all_results['ZeroShot_results'] = zs_results
        with open(eval_args.output_dir+'all_results.json', 'w') as file:
            json.dump(all_results, file, indent=5)

    else:
        ### *** Evaluate FT ***
        torch.cuda.empty_cache()
        last_checkpoint = get_last_checkpoint(eval_args.output_dir)
        logger.info("*** Evaluate FT ***")
        model = OPTForCausalLM.from_pretrained(
                    last_checkpoint,
                    config=config,
                    #torch_dtype=torch.float16,
                    revision=model_args.model_revision,
                    use_auth_token=True if model_args.use_auth_token else None,
                    ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                )
        model = model.to("cuda")

        lm_eval_model = LMEval(model, tokenizer)
        ft_results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=[data_args.dataset_name],
                batch_size=128,
                no_cache=True,
            )    

        # only useful when computing inference memory
        if torch.cuda.is_available() and eval_args.compute_memory:
            peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
            print(
                "Memory utilization",
                peak_memory,
                "GB"
            )

            ft_results['Memory_utilization'] = '{} GB'.format(peak_memory)
        del ft_results['config']['model']

        ### save all results
        with open(eval_args.output_dir+'/all_results.json', 'r') as file:
            all_results = json.load(file)
        all_results['FineTuning_results'] = ft_results
        with open(eval_args.output_dir+'/all_results.json', 'w') as file:
            json.dump(all_results, file, indent=5)
    return 


if __name__ == "__main__":
    main()