import pickle
import torch
import gc



def beam_search_step(batch, tokenizer, model, args):
    # 1 - beam search
    if args.generation_method == "beam_search":
        summary_ids = model.generate(
            batch["text_inputs"]['input_ids'],
            attention_mask=batch["text_inputs"]["attention_mask"],
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            max_length=args.max_summary_length,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=True,
            early_stopping=True
        )
    # 2 - diverse beam search
    if args.generation_method == "diverse_beam_search":
        summary_ids = model.generate(
            batch["text_inputs"]['input_ids'],
            attention_mask=batch["text_inputs"]["attention_mask"],
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            num_return_sequences=args.num_return_sequences,
            max_length=args.max_summary_length,
            diversity_penalty=args.diversity_penalty,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=True,
            early_stopping=True
        )
    # 3 - top-p sampling
    if args.generation_method == "top_p_sampling":
        summary_ids = model.generate(
            batch["text_inputs"]['input_ids'],
            attention_mask=batch["text_inputs"]["attention_mask"],
            num_beams=1,
            do_sample=True,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
            max_length=args.max_summary_length,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=True,
            early_stopping=True
        )
    # 4 - top-k sampling
    if args.generation_method == "top_k_sampling":
        summary_ids = model.generate(
            batch["text_inputs"]['input_ids'],
            attention_mask=batch["text_inputs"]["attention_mask"],
            num_beams=1,
            do_sample=True,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
            max_length=args.max_summary_length,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=True,
            early_stopping=True
        )
    generated = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    del summary_ids
    gc.collect()

    return generated
