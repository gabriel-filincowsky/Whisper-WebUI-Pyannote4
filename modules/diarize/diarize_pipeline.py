# Adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py

import numpy as np
import pandas as pd
import os
from pyannote.audio import Pipeline
from typing import Optional, Union
import torch

from modules.whisper.data_classes import *
from modules.utils.paths import DIARIZATION_MODELS_DIR
from modules.diarize.audio_loader import load_audio, SAMPLE_RATE
from modules.utils.vram_diagnostics import log_vram, log_vram_delta, get_vram_stats


class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-community-1",
        cache_dir: str = DIARIZATION_MODELS_DIR,
        use_auth_token=None,
        token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
        chunk_length: Optional[float] = None,
        overlap_length: Optional[float] = None,
    ):
        """
        Initialize diarization pipeline with pyannote.audio 4.x compatibility.
        
        Parameters
        ----------
        model_name : str
            Hugging Face model identifier
        cache_dir : str
            Directory to cache models
        use_auth_token : str, optional
            Deprecated: Use 'token' instead. Kept for backward compatibility.
        token : str, optional
            Hugging Face authentication token (preferred for pyannote.audio 4.x)
        device : str or torch.device
            Device to run the pipeline on
        chunk_length : float, optional
            Chunk length in seconds. If None, uses model default (10s).
            WARNING: Avoid values between 6-10 seconds as they trigger excessive memory usage.
        overlap_length : float, optional
            Overlap length in seconds between chunks. If None, uses model default.
            Step = chunk_length - overlap_length.
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        
        # pyannote.audio 4.x uses 'token' instead of 'use_auth_token'
        # Support both for compatibility
        auth_token = token if token is not None else use_auth_token
        
        # Try with 'token' first (pyannote.audio 4.x), fall back to 'use_auth_token' if needed
        try:
            self.model = Pipeline.from_pretrained(
                model_name,
                token=auth_token,
                cache_dir=cache_dir
            ).to(device)
        except TypeError:
            # Fallback for older API if needed
            self.model = Pipeline.from_pretrained(
                model_name,
                use_auth_token=auth_token,
                cache_dir=cache_dir
            ).to(device)
        
        # Configure chunk length and step (overlap) if provided
        # This must be done after loading the pipeline
        if chunk_length is not None:
            # Validate chunk_length to avoid memory surge range
            if 6.0 <= chunk_length <= 10.0:
                import warnings
                warnings.warn(
                    f"Chunk length {chunk_length}s is in the problematic range (6-10s) that triggers "
                    f"excessive memory usage (9+ GB). Consider using < 5s or > 11s for optimal memory usage.",
                    UserWarning
                )
            
            self.model._segmentation.duration = chunk_length
            
            # Set step based on overlap_length
            if overlap_length is not None:
                if overlap_length >= chunk_length:
                    raise ValueError(f"Overlap length ({overlap_length}s) must be less than chunk length ({chunk_length}s)")
                step = chunk_length - overlap_length
                self.model._segmentation.step = step
            else:
                # Default to 50% overlap if not specified
                self.model._segmentation.step = chunk_length / 2.0

    def __call__(self, audio: Union[str, np.ndarray], min_speakers=None, max_speakers=None, chunk_length=None, overlap_length=None):
        log_vram("PIPELINE.__call__:start", f"audio_type={type(audio).__name__}")
        pre_stats = get_vram_stats()
        
        if isinstance(audio, str):
            audio = load_audio(audio)
        
        # Move audio tensor to the same device as the model
        audio_tensor = torch.from_numpy(audio[None, :]).to(self.device)
        log_vram_delta("PIPELINE.__call__:audio_to_gpu", pre_stats, f"tensor_shape={audio_tensor.shape}")
        
        audio_data = {
            'waveform': audio_tensor,
            'sample_rate': SAMPLE_RATE
        }
        
        # Configure chunk length and step if provided (allows runtime override)
        if chunk_length is not None:
            if 6.0 <= chunk_length <= 10.0:
                import warnings
                warnings.warn(
                    f"Chunk length {chunk_length}s is in the problematic range (6-10s) that triggers "
                    f"excessive memory usage (9+ GB). Consider using < 5s or > 11s for optimal memory usage.",
                    UserWarning
                )
            self.model._segmentation.duration = chunk_length
            if overlap_length is not None:
                if overlap_length >= chunk_length:
                    raise ValueError(f"Overlap length ({overlap_length}s) must be less than chunk length ({chunk_length}s)")
                step = chunk_length - overlap_length
                self.model._segmentation.step = step
        
        # Ensure model is in eval mode and disable gradient tracking to prevent computation graph retention
        if hasattr(self.model, 'eval'):
            self.model.eval()
        
        # Use torch.no_grad() to prevent PyTorch from maintaining computation graph
        # This is critical for freeing GPU memory after inference
        pre_inference_stats = get_vram_stats()
        log_vram("PIPELINE.__call__:before_inference", "Starting model inference")
        
        with torch.no_grad():
            segments = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
        
        log_vram_delta("PIPELINE.__call__:after_inference", pre_inference_stats, "Model inference complete")
        
        # pyannote.audio 4.x returns DiarizeOutput object, extract Annotation from speaker_diarization
        if hasattr(segments, 'speaker_diarization'):
            log_vram("PIPELINE.__call__:extracting_annotation", "DiarizeOutput detected, extracting speaker_diarization")
            segments = segments.speaker_diarization
        
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        
        log_vram("PIPELINE.__call__:before_cleanup", f"DataFrame created with {len(diarize_df)} segments")
        pre_cleanup_stats = get_vram_stats()
        
        # Aggressively clean up GPU memory
        # Move tensor to CPU before deletion to ensure proper cleanup
        if audio_tensor.is_cuda:
            audio_tensor = audio_tensor.cpu()
        del audio_data['waveform']
        del audio_tensor
        
        # Clear GPU cache and synchronize to ensure cleanup completes
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        log_vram_delta("PIPELINE.__call__:after_cleanup", pre_cleanup_stats, "Audio tensor cleanup complete")
        log_vram_delta("PIPELINE.__call__:end", pre_stats, "Returning DataFrame")
        
        return diarize_df


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    transcript_segments = transcript_result["segments"]
    if transcript_segments and isinstance(transcript_segments[0], Segment):
        transcript_segments = [seg.model_dump() for seg in transcript_segments]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'],
                                                                                            seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])

        intersected = diarize_df[diarize_df["intersection"] > 0]

        speaker = None
        if len(intersected) > 0:
            # Choosing most strong intersection
            speaker = intersected.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
        elif fill_nearest:
            # Otherwise choosing closest
            speaker = diarize_df.sort_values(by=["intersection"], ascending=False)["speaker"].values[0]

        if speaker is not None:
            seg["speaker"] = speaker

        # assign speaker to words
        if 'words' in seg and seg['words'] is not None:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(
                        diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'],
                                                                                                  word['start'])

                    intersected = diarize_df[diarize_df["intersection"] > 0]

                    word_speaker = None
                    if len(intersected) > 0:
                        # Choosing most strong intersection
                        word_speaker = \
                            intersected.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                    elif fill_nearest:
                        # Otherwise choosing closest
                        word_speaker = diarize_df.sort_values(by=["intersection"], ascending=False)["speaker"].values[0]

                    if word_speaker is not None:
                        word["speaker"] = word_speaker

    return {"segments": transcript_segments}


class DiarizationSegment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
