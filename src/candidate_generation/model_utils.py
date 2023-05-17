import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, \
    BartTokenizerFast, BartForConditionalGeneration


def build_tokenizer(args):
    tokenizer = None
    if args.model_type.startswith("pegasus"):
        print("\nUsing Pegasus tokenizer")
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large", cache_dir = args.cache_dir)
    elif args.model_type.startswith("bart"):
        print("\nUsing Bart tokenizer")
        tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-large", cache_dir = args.cache_dir)

    return tokenizer


def build_model(args):
    model = None
    if args.model_type == "pegasus":
        print("\nUsing Pegasus model")
        # checkpoint shared on Huggingface
        if not("our" in args.model_name):
            model = PegasusForConditionalGeneration.from_pretrained(args.model_name, cache_dir = args.cache_dir)
        else:
            model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large", cache_dir=args.cache_dir)
            model.load_state_dict(torch.load(args.load_model_path))
    elif args.model_type == "bart":
        print("\nUsing Bart model")
        if not ("our" in args.model_name):
            model = BartForConditionalGeneration.from_pretrained(args.model_name, cache_dir = args.cache_dir)
        else:
            model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", cache_dir=args.cache_dir)
            model.load_state_dict(torch.load(args.load_model_path))

    return model
