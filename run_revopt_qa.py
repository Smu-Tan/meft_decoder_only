import os
import sys
import math
import torch
import logging
import transformers

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    DataCollatorForTokenClassification,
    set_seed,
    EarlyStoppingCallback
)

from data.data import get_raw_datasets, process_text2text_datasets
from utils.utils import ModelArguments, DataTrainingArguments, TrainingArguments

from models import OPTForCausalLM
from trainers import CustomTrainer

#### preparations
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

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

    model = OPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                #torch_dtype=torch.float16,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
    logger.info(model)
    logger.info(f"Total num of parameters: {sum(p.numel() for p in model.parameters())}")

    if model_args.adapter_bottleneck_dim > 0:
        for param in model.parameters():
            param.requires_grad = False
        non_freeze_sets = ["adapter", "concat_layer"]
        if model_args.freeze_irreversible_layers:
            non_freeze_sets.append(f"decoder.layers.{config.num_hidden_layers - model_args.num_rev_layers - 1}."
                               f"fc2.bias")
        else:
            non_freeze_sets.append("embed_positions")
        for key in non_freeze_sets:
            for n, p in model.named_parameters():
                if key in n:
                    p.requires_grad = True
    
    raw_datasets = get_raw_datasets(data_args)
    tokenized_datasets = process_text2text_datasets(raw_datasets, data_args, model_args, tokenizer)
    if training_args.do_train:
        train_dataset = tokenized_datasets['train']

    if training_args.do_eval:
        eval_dataset = tokenized_datasets['validation']

    if training_args.do_predict:
        test_dataset = tokenized_datasets['test']

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, max_length=data_args.max_seq_length)

    if model_args.freeze_irreversible_layers:
        training_args.start_layer = config.num_hidden_layers - model_args.num_rev_layers - 1
    else:
        training_args.start_layer = -1

    trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset= eval_dataset if training_args.do_eval else None,
            #compute_metrics=compute_metrics, #if training_args.predict_with_generate else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]
            #preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
    
    performance_metrics = {}
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        #if training_args.compute_time:
        torch.cuda.synchronize()  # wait for move to complete
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        performance_metrics.update({"mem(G) before training": torch.cuda.memory_allocated() / (1024 * 1024 * 1000)})
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        performance_metrics.update({"mem(G) after training": torch.cuda.memory_allocated() / (1024 * 1024 * 1000)})

        #if training_args.compute_time:
        end.record()
        torch.cuda.synchronize()  # wait for all_reduce to complete
        total_time = start.elapsed_time(end)/(1000*60)
        performance_metrics.update({"total_time in minutes ": total_time})

        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        #metrics["mem(G)"] = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1000)


        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        performance_metrics.update({"peak mem(G)": torch.cuda.max_memory_allocated() / 1024 ** 2 / 1000})
        #trainer.save_metrics("performance", performance_metrics)

        if torch.cuda.is_available() and training_args.compute_memory:
            peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
            print(
                "Memory utilization",
                peak_memory,
                "GB"
            )
            performance_metrics.update({"peak_memory": peak_memory})
    #if training_args.compute_memory or training_args.compute_time:
    #print(performance_metrics)
    trainer.save_metrics("performance", performance_metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        ppl = math.exp(metrics['eval_loss'])
        metrics['ppl']=ppl
        print(f"Perplexity: {math.exp(metrics['eval_loss']):.2f}")
        metrics["mem(G)"] = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1000)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        performance_metrics = {"Mem(G)": torch.cuda.max_memory_allocated() / 1024 ** 2 / 1000}
        trainer.save_metrics("performance", performance_metrics)

        # only useful when computing inference memory
        if torch.cuda.is_available() and training_args.compute_memory:
            peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
            print(
                "Memory utilization",
                peak_memory,
                "GB"
            )
    return


if __name__ == "__main__":
    main()