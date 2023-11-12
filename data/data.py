from datasets import load_dataset
import logging
from itertools import chain
from data.tasks import task_dict, map_dataset_name_and_config

logger = logging.getLogger(__name__)


def get_raw_datasets(args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset_name, dataset_config_name = map_dataset_name_and_config(args)
        raw_datasets = load_dataset(
            dataset_name, dataset_config_name, cache_dir=args.data_cache_dir)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.data_cache_dir
            )
            raw_datasets["train"] = load_dataset(
                dataset_name,
                dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.data_cache_dir
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        elif extension == 'zst':
            extension = 'json'
        raw_datasets = load_dataset(
            extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    return raw_datasets


def get_tokenized_datasets(raw_datasets, args, accelerator, tokenizer, lm_type='clm'):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        if lm_type == 'clm':
            return tokenizer(examples[text_column_name])
        elif lm_type == 'mlm':
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
        else:
            raise ValueError(f'lm_type {lm_type} not supported')

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    return tokenized_datasets


def _get_block_size(args, tokenizer):
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)
    return block_size


def process_text2text_datasets(raw_datasets, data_args, model_args,  tokenizer):
    task = task_dict[data_args.dataset_name]

    column_names = raw_datasets["train"].column_names

    def tokenize_function(examples):
        context = task.get_context(examples)
        target = task.get_target(examples)

        #shaomu add space after context
        #context = [i + ' ' for i in context]

        context = tokenizer(context, padding=False, truncation=True, max_length=data_args.max_seq_length)
        target = tokenizer(target, padding=False, truncation=True, max_length=data_args.max_seq_length)

        # if context is ending with special token, remove it
        if len(context['input_ids'][0]) > 0 and context['input_ids'][0][-1] in tokenizer.all_special_ids:
            context['input_ids'] = [i[:-1] for i in context['input_ids']]
            context['attention_mask'] = [a[:-1]
                                         for a in context['attention_mask']]

        # if target is starting with special token, remove it
        if len(target['input_ids'][0]) > 0 and target['input_ids'][0][0] in tokenizer.all_special_ids:
            target['input_ids'] = [i[1:] for i in target['input_ids']]
            target['attention_mask'] = [a[1:]
                                        for a in target['attention_mask']]


        out = {}
        out['input_ids'] = [i1 + i2 for i1,
                            i2 in zip(context['input_ids'], target['input_ids'])]
        out['attention_mask'] = [a1 + a2 for a1,
                                 a2 in zip(context['attention_mask'], target['attention_mask'])]

        # set -100 for context tokens
        out["labels"] = [
            [-100] * len(i1) + i2 for i1, i2 in zip(context['input_ids'], target['input_ids'])]

        return out

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        #preprocessing_num_workers
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset")

    return tokenized_datasets