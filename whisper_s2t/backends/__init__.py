import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

from ..configs import *
from ..data import WhisperDataLoader
from ..audio import LogMelSpectogram
from ..speech_segmenter import SpeechSegmenter


class NoneTokenizer:
    def __init__(self):
        self.sot_prev = 0
        self.silent_token = 0
        self.no_timestamps = 0
        self.timestamp_begin = 0
    
    def sot_sequence(self, task=None, lang=None):
        return [task, lang]

    def encode(self, text):
        return [0]


def fix_batch_param(param, default_value, N):
    if param is None:
        param = N*[default_value]
    elif type(param) == type(default_value):
        param = N*[param]
    elif len(param) != N:
        param = N*[param[0]]

    return param


class WhisperModel(ABC):
    def __init__(self,
                 tokenizer=None,
                 vad_model=None,
                 n_mels=80,
                 device="cuda",
                 device_index=0,
                 compute_type="float16",
                 merge_chunks=True,
                 dta_padding=3.0,
                 use_dynamic_time_axis = False,
                 max_speech_len=29.0,
                 max_text_token_len=MAX_TEXT_TOKEN_LENGTH,
                 without_timestamps=True,
                 speech_segmenter_options={}):
        
        # Configure Params
        self.device = device
        self.device_index = device_index
        self.compute_type = compute_type

        self.n_mels = n_mels
        self.merge_chunks = merge_chunks
        self.max_speech_len = max_speech_len

        self.dta_padding = dta_padding
        self.use_dynamic_time_axis = use_dynamic_time_axis

        self.without_timestamps = without_timestamps
        self.max_text_token_len = max_text_token_len

        self.vad_model = vad_model
        self.speech_segmenter_options = speech_segmenter_options
        self.speech_segmenter_options['max_seg_len'] = self.max_speech_len

        # Tokenizer
        if tokenizer is None:
            tokenizer = NoneTokenizer()

        self.tokenizer = tokenizer

        self._init_dependables()


    def _init_dependables(self):
        # Rescaled Params
        self.dta_padding = int(self.dta_padding*SAMPLE_RATE)
        self.max_initial_prompt_len = 218

        # Load Pre Processor
        self.preprocessor = LogMelSpectogram(n_mels=self.n_mels).to(self.device)
        self.align_preprocessor = LogMelSpectogram(n_mels=80).to(self.device)

        # Load Speech Segmenter
        self.speech_segmenter = SpeechSegmenter(self.vad_model, device=self.device, **self.speech_segmenter_options)

        # Load Data Loader
        self.data_loader = WhisperDataLoader(
            self.device, self.tokenizer, self.speech_segmenter, 
            dta_padding=self.dta_padding,
            without_timestamps=self.without_timestamps, 
            max_speech_len=self.max_speech_len, 
            max_initial_prompt_len=self.max_initial_prompt_len, 
            use_dynamic_time_axis=self.use_dynamic_time_axis,
            merge_chunks=self.merge_chunks
        )

    def update_params(self, params={}):
        for key, value in params.items():
            setattr(self, key, value)
        
        self._init_dependables()

    
    @abstractmethod
    def generate_segment_batched(self, features, prompts):
        pass
        
    @torch.no_grad()
    def transcribe(self, audio_files, lang_codes=None, tasks=None, initial_prompts=None, batch_size=8, word_timestamps=True, without_timestamps=True, generation_kwargs={}):
        
        # if lang_codes == None:
        #     lang_codes = len(audio_files)*['en']
            
        # if tasks == None:
        #     tasks = len(audio_files)*['transcribe']
        
        # if initial_prompts == None:
        #     initial_prompts = len(audio_files)*[None]
            
        # responses = []
        # for signals, prompts, seq_len in self.data_loader(audio_files, lang_codes, tasks, initial_prompts, batch_size=batch_size, use_vad=False):
        #     mels, seq_len = self.preprocessor(signals, seq_len)
        #     res = self.generate_segment_batched(mels.to(self.device), prompts)
        #     responses.extend(res)
        
        # return responses

        lang_codes = fix_batch_param(lang_codes, 'en', len(audio_files))
        tasks = fix_batch_param(tasks, 'transcribe', len(audio_files))
        initial_prompts = fix_batch_param(initial_prompts, None, len(audio_files))
            
        responses = [[] for _ in audio_files]
        
        pbar_pos = 0
        with tqdm(total=len(audio_files)*100, desc=f"Transcribing") as pbar:
            for signals, prompts, seq_len, seg_metadata, pbar_update in self.data_loader(audio_files, lang_codes, tasks, initial_prompts, batch_size=batch_size, without_timestamps=without_timestamps, use_vad=False):
                mels, main_seq_len = self.preprocessor(signals, seq_len)
                align_mels, align_seq_len = self.align_preprocessor(signals, seq_len) if word_timestamps else (None, None)
                res = self.generate_segment_batched(mels.to(self.device), prompts, main_seq_len, seg_metadata, align_mels.to(self.device) if align_mels is not None else None, align_seq_len, generation_kwargs)

                for segment in res:
                    try:
                        start_time = round(segment['word_timestamps'][0]['start'], 3) if word_timestamps else segment['start_time']
                        end_time = round(segment['word_timestamps'][-1]['end'], 3) if word_timestamps else segment['end_time']
                    except:
                        print("segment", segment)
                        raise
                    responses[0].append({**segment,
                                         'start_time': start_time,
                                         'end_time': end_time})
                
                if (pbar_pos) <= pbar.total:
                    pbar_pos += pbar_update
                    pbar.update(pbar_update)
            
            pbar.update(pbar.total-pbar_pos)
        
        return responses

    @torch.no_grad()
    def transcribe_with_vad(self, audio_files, lang_codes=None, tasks=None, initial_prompts=None, batch_size=8, word_timestamps=True, without_timestamps=True, generation_kwargs={}):

        lang_codes = fix_batch_param(lang_codes, 'en', len(audio_files))
        tasks = fix_batch_param(tasks, 'transcribe', len(audio_files))
        initial_prompts = fix_batch_param(initial_prompts, None, len(audio_files))
            
        responses = [[] for _ in audio_files]
        
        pbar_pos = 0
        with tqdm(total=len(audio_files)*100, desc=f"Transcribing") as pbar:
            for signals, prompts, seq_len, seg_metadata, pbar_update in self.data_loader(audio_files, lang_codes, tasks, initial_prompts, batch_size=batch_size, without_timestamps=without_timestamps):
                mels, main_seq_len = self.preprocessor(signals, seq_len)
                align_mels, align_seq_len = self.align_preprocessor(signals, seq_len) if word_timestamps else (None, None)
                res = self.generate_segment_batched(mels.to(self.device), prompts, main_seq_len, seg_metadata, align_mels.to(self.device) if align_mels is not None else None, align_seq_len, generation_kwargs)

                for segment in res:
                    start_time = round(segment['word_timestamps'][0]['start'], 3) if word_timestamps and len(segment['word_timestamps']) else segment['start_time']
                    end_time = round(segment['word_timestamps'][-1]['end'], 3) if word_timestamps and len(segment['word_timestamps']) else segment['end_time']
                    responses[0].append({**segment,
                                         'start_time': start_time,
                                         'end_time': end_time})
                
                if (pbar_pos) <= pbar.total:
                    pbar_pos += pbar_update
                    pbar.update(pbar_update)
            
            pbar.update(pbar.total-pbar_pos)
        
        return responses