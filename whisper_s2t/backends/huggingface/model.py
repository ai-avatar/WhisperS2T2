import torch
import numpy as np

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import ctranslate2

from ..ctranslate2.hf_utils import download_model
from .. import WhisperModel
from ...configs import *


ASR_OPTIONS = {
    "beam_size": 1,
    "without_timestamps": True,
    "return_scores": False,
    "return_no_speech_prob": False,
    "use_flash_attention": True,
    "use_better_transformer": False,
    "aligner_model_instance": None,
}


COMPUTE_TYPE_TO_TORCH_DTYPE = {
    "float16": torch.float16
}

TOKEN_TIMESTAMP_BEGIN = 50365
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
                                                                     use_flash_attention_2=self.asr_options["use_flash_attention"])
        self.model.config.forced_decoder_ids = None
        self.model.to(device).eval()

        if self.asr_options["use_better_transformer"]:
            self.model = self.model.to_bettertransformer()

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
        }

        super().__init__(
            device=device,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs
        )

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
            res = self.aligner_model.align(ctranslate2.StorageView.from_array(features[req_idx]), 
                                           start_sequence=list(start_seq), 
                                           text_tokens=[text_tokens[_] for _ in req_idx],
                                           num_frames=list(seq_lens[req_idx].detach().cpu().numpy()), 
                                           median_filter_width=7)

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
    
    def generate_segment_batched(self, features, prompts, seq_lens, seg_metadata, align_features, align_seq_lens):
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
            result = self.model.generate(features[idx_list], 
                                                task=task,
                                                language=lang,
                                                **self.generate_kwargs)
        print(result)
        # group tokens by utterance (separated by timestamp tokens)
        tokens = [[]]
        group = 0
        groups_per_segment = []
        group_timestamps = []
        for i, segment in enumerate(result):
            for token in segment:
                if token > TOKEN_TIMESTAMP_BEGIN and len(tokens[group]):
                    tokens.append([])
                    groups_per_segment.append(len(tokens[group]))
                    group += 1
                elif token < TOKEN_EOT:
                    tokens[group].append(token)

                if token >= TOKEN_TIMESTAMP_BEGIN:
                    group_timestamps.append((token - TOKEN_TIMESTAMP_BEGIN) * TIME_PRECISION)
            
            if len(group_timestamps) == 0:
                group_timestamps.append(round(seg_metadata[i]['start_time'], 3))

            # fallback to segment end_time if end time was not predicted
            if len(group_timestamps) % 2 == 1:
                group_timestamps.append(round(seg_metadata[i]['end_time'], 3))

        if len(tokens[-1]) == 0:
            tokens = tokens[:-1]

        text_groups = self.processor.batch_decode(tokens)

        texts = []
        for idx, num_groups in enumerate(groups_per_segment):
            texts.append(" ".join(text_groups[idx:num_groups+idx]))
        
        response = []
        for idx, r in enumerate(text_groups):
            response.append({'text': text_groups[idx].strip(),
                             'start_time': group_timestamps[idx*2],
                             'end_time': group_timestamps[idx*2+1]})

        if align_features is not None:
            text_tokens = [x + [TOKEN_EOT] for x in result]
            sot_seqs = [tuple(_[-4:]) for _ in prompts]
            word_timings = self.align_words(align_features, texts, text_tokens, sot_seqs, align_seq_lens, seg_metadata)

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