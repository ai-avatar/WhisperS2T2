import torch
import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.utils import is_flash_attn_2_available
import ctranslate2
import threading
from contextlib import contextmanager

from ..ctranslate2.hf_utils import download_model
from .. import WhisperModel
from ...configs import *
from .tokenizer import Tokenizer


ASR_OPTIONS = {
    "beam_size": 1,
    "without_timestamps": True,
    "return_scores": False,
    "return_no_speech_prob": False,
    "use_flash_attention": True, # deprecated
    "use_better_transformer": False, # deprecated
    "word_aligner_model": "tiny",
    "aligner_model_instance": None,
    "torch_compile": False,
}


COMPUTE_TYPE_TO_TORCH_DTYPE = {
    "float16": torch.float16
}

TOKEN_TIMESTAMP_BEGIN = 50364
TOKEN_EOT = 50257


class WhisperModelHF(WhisperModel):
    def __init__(self,
                 model_name: str,
                 device="cuda",
                 compute_type="float16",
                 max_text_token_len=MAX_TEXT_TOKEN_LENGTH,
                 asr_options={},
                 **model_kwargs):

        self.model_name = model_name
        self.asr_options = ASR_OPTIONS
        self.asr_options.update(asr_options)

        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name, 
                                                                     torch_dtype=COMPUTE_TYPE_TO_TORCH_DTYPE.get(compute_type, torch.float32), 
                                                                     low_cpu_mem_usage=True, 
                                                                     use_safetensors=True,
                                                                     attn_implementation=("flash_attention_2" if is_flash_attn_2_available() else "sdpa"))
        self.model.config.forced_decoder_ids = None
        self.model.to(device)

        def compile_forward_fn(forward_fn, **compile_kwargs):
            compiled_forward = torch.compile(forward_fn, **compile_kwargs)
            def wrapped_forward(*args, **kwargs):
                if self.model.use_compiled.value:
                    return compiled_forward(*args, **kwargs)
                return forward_fn(*args, **kwargs)
            return wrapped_forward

        self.model.use_compiled = threading.local()
        self.model.use_compiled.value = True
        if self.asr_options["torch_compile"]:
            self.model.generation_config.cache_implementation = "static"
            self.model.forward = compile_forward_fn(self.model.forward, mode="reduce-overhead", fullgraph=True)

        if self.asr_options["aligner_model_instance"]:
            self.aligner_model = self.asr_options["aligner_model_instance"]
        else:
            self.aligner_model_path = download_model(self.asr_options['word_aligner_model'])
            self.aligner_model = ctranslate2.models.Whisper(self.aligner_model_path,
                                                            device=device,
                                                            device_index=0,
                                                            compute_type=compute_type,
                                                            intra_threads=2,
                                                            inter_threads=1)
        
        self.generate_kwargs = {
            "max_new_tokens": max_text_token_len,
            "num_beams": self.asr_options['beam_size'],
            "return_timestamps": not self.asr_options['without_timestamps'],
            "output_scores": True,
            "return_dict_in_generate": True,
        }

        tokenizer = Tokenizer(self.processor.tokenizer)

        super().__init__(
            tokenizer=tokenizer,
            device=device,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs
        )

    @contextmanager
    def use_torch_compile(self, value: bool):
        self.model.use_compiled.value = value
        try:
            yield
        finally:
            self.model.use_compiled.value = True

    # deprecated
    def update_generation_kwargs(self, params={}):
        self.generate_kwargs.update(params)

        if 'max_new_tokens' in params:
            self.update_params(params={'max_text_token_len': params['max_new_tokens']})
    
    def assign_word_timings(self, alignments, text_token_probs, words, word_tokens):
        text_indices = np.array([pair[0] for pair in alignments])
        time_indices = np.array([pair[1] for pair in alignments])
    
        if len(word_tokens) <= 1:
            return []
            
        word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
        if len(word_boundaries) <= 1:
            return []
    
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps]*TIME_PRECISION
        start_times = jump_times[word_boundaries[:-1]]
        end_times = jump_times[word_boundaries[1:]]
        word_probs = [
            np.mean(text_token_probs[i:j])
            for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
        ]
    
        return [
            dict(
                word=word, start=round(start, 2), end=round(end, 2), prob=round(prob, 2)
            )
            for word, start, end, prob in zip(
                words, start_times, end_times, word_probs
            )
        ]

    def align_words(self, features, texts, text_tokens, sot_seqs, seq_lens, seg_metadata):
        lang_codes = [_['lang_code'] for _ in seg_metadata]
        word_tokens = self.tokenizer.split_to_word_tokens_batch(texts, text_tokens, lang_codes)

        if len(word_tokens) == 0:
            return []

        start_seq_wise_req = {}
        for _idx, _sot_seq in enumerate(sot_seqs):
            try:
                # print(_sot_seq)
                start_seq_wise_req[_sot_seq].append(_idx)
            except:
                start_seq_wise_req[_sot_seq] = [_idx]

        token_alignments = [[] for _ in seg_metadata]
        for start_seq, req_idx in start_seq_wise_req.items():
            try:
                res = self.aligner_model.align(ctranslate2.StorageView.from_array(features[req_idx]), 
                                            start_sequence=list(start_seq), 
                                            text_tokens=[text_tokens[_] for _ in req_idx],
                                            num_frames=list(seq_lens[req_idx].detach().cpu().numpy()), 
                                            median_filter_width=7)
            except Exception as e:
                print("Error in aligning words:")
                print("features:", features[req_idx])
                print("feature shape:", features[req_idx].shape)
                print("start_seq:", start_seq)
                print("text_tokens:", [text_tokens[_] for _ in req_idx])
                print("num_frames:", list(seq_lens[req_idx].detach().cpu().numpy()))
                raise e

            for _res, _req_idx in zip(res, req_idx):
                token_alignments[_req_idx] = _res

        word_timings = []
        for _idx, _seg_metadata in enumerate(seg_metadata):
            if _idx < len(word_tokens) and len(word_tokens[_idx]) >= 2:
                align_words = word_tokens[_idx][0]
                align_word_tokens = word_tokens[_idx][1]
            else:
                print("No word tokens found for segment", _idx)
                print("Segment Texts:", texts)
                print("word_tokens:", word_tokens)
                print("meta:", seg_metadata)
                align_words = []
                align_word_tokens = []

            _word_timings = self.assign_word_timings(token_alignments[_idx].alignments, 
                                                    token_alignments[_idx].text_token_probs, 
                                                    align_words,
                                                    align_word_tokens)
        
            stitched_seg = _seg_metadata['stitched_seg']

            current_seg_idx = 0
            current_offset = _seg_metadata['start_time']
        
            for w in _word_timings:
                while (w['start'] + current_offset) >= stitched_seg[current_seg_idx][1]:
                    current_seg_idx += 1
                    current_offset += (stitched_seg[current_seg_idx][0]-stitched_seg[current_seg_idx-1][1])
        
                w['start'] += current_offset
                w['end'] += current_offset
        
            word_timings.append(_word_timings)

        return word_timings

    def generate_segment_batched(self, features, prompts, seq_lens, seg_metadata, align_features, align_seq_lens, generation_kwargs={}):
        if self.compute_type == "float16":
            features = features.to(self.device).half()

        lang_and_task_pairs = {}
        for _i, _p in enumerate(prompts):
            try:
                lang_and_task_pairs[(_p[-3], _p[-2])].append(_i)
            except:
                lang_and_task_pairs[(_p[-3], _p[-2])] = [_i]

        response = [{} for _ in prompts]
        for (task, lang), idx_list in lang_and_task_pairs.items():
            has_prompt = 'prompt_ids' in generation_kwargs and generation_kwargs['prompt_ids'] is not None
            # disable torch compile if prompt or custom logits_processor is present to avoid recompilation
            use_torch_compile = not has_prompt and ('logits_processor' not in generation_kwargs or generation_kwargs['logits_processor'] is None or generation_kwargs['logits_processor'] == [])
            with self.use_torch_compile(use_torch_compile):
                generate_result = self.model.generate(features[idx_list], 
                                                    task=task,
                                                    language=lang,
                                                    **(self.generate_kwargs | generation_kwargs))
            logprobs = self.model.compute_transition_scores(generate_result["sequences"], generate_result["scores"], normalize_logits=True)
            result = generate_result["sequences"]
            # remove prompt tokens from the result
            if has_prompt:
                result = [segment[len(generation_kwargs['prompt_ids']):] for segment in result]

        # group tokens by utterance (separated by timestamp tokens)
        tokens = [[]]
        group_idx = 0
        group_timestamps = [round(seg_metadata[0]['start_time'], 3), round(seg_metadata[0]['end_time'], 3)]
        group_logprobs = [[]]
        for i, segment in enumerate(result):
            logprobs_segment = logprobs[i]
            token_without_logprobs = len(segment) - len(logprobs_segment)
            for token_i, token in enumerate(segment):
                if token_i > token_without_logprobs and token < TOKEN_EOT:
                    group_logprobs[group_idx].append(logprobs_segment[token_i - token_without_logprobs])

                if token > TOKEN_TIMESTAMP_BEGIN and len(tokens[group_idx]):
                    tokens.append([])
                    group_timestamps[group_idx*2+1] = (token - TOKEN_TIMESTAMP_BEGIN) * TIME_PRECISION
                    group_logprobs.append([])
                    group_idx += 1

                    # set fallback timestamps for new group in case of missing timestamps
                    group_timestamps.append(round(seg_metadata[i]['start_time'], 3))
                    group_timestamps.append(round(seg_metadata[i]['end_time'], 3))
                elif token < TOKEN_EOT:
                    tokens[group_idx].append(token)
                elif token >= TOKEN_TIMESTAMP_BEGIN:
                    # start timestamp found
                    group_timestamps[group_idx*2] = (token - TOKEN_TIMESTAMP_BEGIN) * TIME_PRECISION

        # remove last empty token list
        if len(tokens[-1]) == 0:
            tokens = tokens[:-1]

        text_groups = self.processor.batch_decode(tokens)
        
        response = []
        for idx, r in enumerate(text_groups):
            response.append({'text': text_groups[idx].strip(),
                            'start_time': float(group_timestamps[idx*2]),
                            'end_time': float(group_timestamps[idx*2+1]),
                            'avg_logprob': torch.mean(torch.stack(group_logprobs[idx])).item()})

        if align_features is not None:
            text_tokens = [x.tolist() + [TOKEN_EOT] for x in result]
            sot_seqs = [tuple(self.tokenizer.align_sot_sequence(prompt[0], prompt[1])) for prompt in prompts]
            word_timings = self.align_words(align_features, text_groups, text_tokens, sot_seqs, align_seq_lens, seg_metadata)

            offset = 0
            flat_word_timings = [word for sublist in word_timings for word in sublist]
            for idx, segment in enumerate(response):
                segment_length = len(segment['text'].replace(" ", ""))
                words = []
                empty_words = 0
                for word_timing in flat_word_timings[offset:]:
                    if word_timing['word'] == '':
                        empty_words += 1
                        continue
                    words.append(word_timing)
                    segment_length -= len(word_timing['word'])
                    if segment_length <= 0:
                        offset += len(words) + empty_words
                        break
                segment['word_timestamps'] = words

        return response