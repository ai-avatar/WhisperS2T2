import os
import tokenizers
import ctranslate2
import numpy as np
import torch
from .tokenizer import Tokenizer
from .hf_utils import download_model


from .. import WhisperModel
from ...configs import *


FAST_ASR_OPTIONS = {
    "beam_size": 1,
    "best_of": 1, # Placeholder
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4, # Placeholder
    "log_prob_threshold": -1.0, # Placeholder
    "no_speech_threshold": 0.5, # Placeholder
    "prefix": None, # Placeholder
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False, # Placeholder
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
    "word_aligner_model": 'tiny',
    "aligner_model_instance": None,
    "torch_compile": False,
}


BEST_ASR_CONFIG = {
    "beam_size": 5,
    "best_of": 1, # Placeholder
    "patience": 2,
    "length_penalty": 1,
    "repetition_penalty": 1.01,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4, # Placeholder
    "log_prob_threshold": -1.0, # Placeholder
    "no_speech_threshold": 0.5, # Placeholder
    "prefix": None, # Placeholder
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 1.0,
    "word_timestamps": False, # Placeholder
    "sampling_temperature": 1.0,
    "return_scores": True,
    "return_no_speech_prob": True,
    "word_aligner_model": 'tiny',
    "aligner_model_instance": None,
    "torch_compile": False,
}


class WhisperModelCT2(WhisperModel):
    def __init__(self,
                 model_name_or_path: str,
                 cpu_threads=4,
                 num_workers=1,
                 device="cuda",
                 device_index=0,
                 compute_type="float16",
                 max_text_token_len=MAX_TEXT_TOKEN_LENGTH,
                 asr_options={},
                 **model_kwargs):

        
        # Get local model path or download from huggingface
        if os.path.isdir(model_name_or_path):
            self.model_path = model_name_or_path
        else:
            self.model_path = download_model(model_name_or_path)
        
        # Load model
        self.model = ctranslate2.models.Whisper(self.model_path,
                                                device=device,
                                                device_index=device_index,
                                                compute_type=compute_type,
                                                intra_threads=cpu_threads,
                                                inter_threads=num_workers)
        
        # Load tokenizer
        tokenizer_file = os.path.join(self.model_path, "tokenizer.json")
        tokenizer = Tokenizer(tokenizers.Tokenizer.from_file(tokenizer_file), self.model.is_multilingual)

        # ASR Options
        self.asr_options = FAST_ASR_OPTIONS
        self.asr_options.update(asr_options)

        if self.asr_options["aligner_model_instance"]:
            self.aligner_model = self.asr_options["aligner_model_instance"]
        else:
            self.aligner_model_path = download_model(self.asr_options['word_aligner_model'])
            self.aligner_model = ctranslate2.models.Whisper(self.aligner_model_path,
                                                            device=device,
                                                            device_index=device_index,
                                                            compute_type=compute_type,
                                                            intra_threads=cpu_threads,
                                                            inter_threads=num_workers)
        
        self.generate_kwargs = {
            "max_length": max_text_token_len,
            "return_scores": self.asr_options['return_scores'],
            "return_no_speech_prob": self.asr_options['return_no_speech_prob'],
            "length_penalty": self.asr_options['length_penalty'],
            "repetition_penalty": self.asr_options['repetition_penalty'],
            "no_repeat_ngram_size": self.asr_options['no_repeat_ngram_size'],
            "beam_size": self.asr_options['beam_size'],
            "patience": self.asr_options['patience'],
            "suppress_blank": self.asr_options['suppress_blank'],
            "suppress_tokens": self.asr_options['suppress_tokens'],
            "max_initial_timestamp_index": int(round(self.asr_options['max_initial_timestamp']/TIME_PRECISION)),
            "sampling_temperature": self.asr_options['sampling_temperature'],
        }

        super().__init__(
            tokenizer=tokenizer,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            max_text_token_len=max_text_token_len,
            **model_kwargs
        )

    def update_generation_kwargs(self, params={}):
        self.generate_kwargs.update(params)

        if 'max_text_token_len' in params:
            self.update_params(params={'max_text_token_len': params['max_text_token_len']})
    
    def encode(self, features):
        """
        [Not Used]
        """
        
        features = ctranslate2.StorageView.from_array(features.contiguous())
        return self.model.encode(features)

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
    
    def generate_segment_batched(self, features, prompts, seq_lens, seg_metadata, align_features, align_seq_lens, generation_kwargs={}):
        
        if self.device == 'cpu':
            features = np.ascontiguousarray(features.detach().numpy())
        else:
            features = features.contiguous()

        # returns WhisperGenerationResult (https://github.com/OpenNMT/CTranslate2/blob/617405f4b050e994e829d527da6caa0e0030afe7/include/ctranslate2/models/whisper.h#L61-L75)
        result = self.model.generate(ctranslate2.StorageView.from_array(features),
                                     prompts,
                                     return_logits_vocab=True,
                                     **(self.generate_kwargs | generation_kwargs))


        # group tokens by utterance (separated by timestamp tokens)
        tokens = [[]]
        group = 0
        groups_per_segment = []
        group_timestamps = []
        group_logprobs = [[]]
        for i, segment in enumerate(result):
            # Calculate log probabilities from logits
            logits = []
            for logit_array in segment.logits:
                for logit in logit_array:
                    logits.append(torch.tensor(np.array(logit.to_device(ctranslate2.Device(0)))))
            
            # Stack logits into a single tensor before applying softmax
            logits_tensor = torch.stack(logits)
            probs = torch.nn.functional.softmax(logits_tensor, dim=-1)
            log_probs = torch.log(probs)
            
            # Get log probability for the predicted token
            token_log_probs = []
            for j, token_id in enumerate(segment.sequences_ids[0]):
                if j < len(log_probs):
                    token_log_probs.append(log_probs[j][token_id].item())

            for idx, token in enumerate(segment.sequences_ids[0]):
                if token > self.tokenizer.timestamp_begin and len(tokens[group]):
                    tokens.append([])
                    group_logprobs.append([])
                    groups_per_segment.append(len(tokens[group]))
                    group += 1
                elif token < self.tokenizer.eot:
                    tokens[group].append(token)
                    group_logprobs[group].append(token_log_probs[idx])
                
                if token >= self.tokenizer.timestamp_begin:
                    group_timestamps.append((token - self.tokenizer.timestamp_begin) * TIME_PRECISION)
            
            if len(group_timestamps) == 0:
                group_timestamps.append(round(seg_metadata[i]['start_time'], 3))

            # fallback to segment end_time if end time was not predicted
            if len(group_timestamps) % 2 == 1:
                group_timestamps.append(round(seg_metadata[i]['end_time'], 3))

        if len(tokens[-1]) == 0:
            tokens = tokens[:-1]

        text_groups = self.tokenizer.decode_batch(tokens)

        texts = []
        for idx, num_groups in enumerate(groups_per_segment):
            texts.append(" ".join(text_groups[idx:num_groups+idx]))
        
        response = []
        for idx, r in enumerate(text_groups):
            response.append({'text': text_groups[idx].strip(),
                             'start_time': float(group_timestamps[idx*2]),
                             'end_time': float(group_timestamps[idx*2+1]),
                             'avg_logprob': torch.tensor(group_logprobs[idx]).mean().item()})

        if align_features is not None:
            text_tokens = [x.sequences_ids[0]+[self.tokenizer.eot] for x in result]
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