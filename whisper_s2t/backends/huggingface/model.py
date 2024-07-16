import torch

from transformers import WhisperProcessor, WhisperForConditionalGeneration

from .. import WhisperModel
from ...configs import *


ASR_OPTIONS = {
    "beam_size": 1,
    "without_timestamps": True,
    "return_scores": False,
    "return_no_speech_prob": False,
    "use_flash_attention": True,
    "use_better_transformer": False,
}


COMPUTE_TYPE_TO_TORCH_DTYPE = {
    "float16": torch.float16
}


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
            for token in segment.sequences_ids[0]:
                if token > self.tokenizer.timestamp_begin and len(tokens[group]):
                    tokens.append([])
                    groups_per_segment.append(len(tokens[group]))
                    group += 1
                elif token < self.tokenizer.eot:
                    tokens[group].append(token)

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
                             'start_time': group_timestamps[idx*2],
                             'end_time': group_timestamps[idx*2+1]})

        # TODO: implement align_words for HF models
        # if align_features is not None:
        #     text_tokens = [x.sequences_ids[0]+[self.tokenizer.eot] for x in result]
        #     sot_seqs = [tuple(_[-4:]) for _ in prompts]
        #     word_timings = self.align_words(align_features, texts, text_tokens, sot_seqs, align_seq_lens, seg_metadata)

        #     offset = 0
        #     flat_word_timings = [word for sublist in word_timings for word in sublist]
        #     for idx, segment in enumerate(response):
        #         segment_length = len(segment['text'].replace(" ", ""))
        #         words = []
        #         empty_words = 0
        #         for word_timing in flat_word_timings[offset:]:
        #             if word_timing['word'] == '':
        #                 empty_words += 1
        #                 continue
        #             words.append(word_timing)
        #             segment_length -= len(word_timing['word'])
        #             if segment_length <= 0:
        #                 offset += len(words) + empty_words
        #                 break
        #         segment['word_timestamps'] = words

        return response