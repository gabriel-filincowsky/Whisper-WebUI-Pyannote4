import os
import torch
from typing import List, Union, BinaryIO, Optional, Tuple
import numpy as np
import time
import logging
import gc

from modules.utils.paths import DIARIZATION_MODELS_DIR
from modules.diarize.diarize_pipeline import DiarizationPipeline, assign_word_speakers
from modules.diarize.audio_loader import load_audio
from modules.whisper.data_classes import *
from modules.utils.vram_diagnostics import log_vram, log_vram_delta, get_vram_stats


class Diarizer:
    def __init__(self,
                 model_dir: str = DIARIZATION_MODELS_DIR
                 ):
        self.device = self.get_device()
        self.available_device = self.get_available_device()
        self.compute_type = "float16"
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.pipe = None

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            transcribed_result: List[Segment],
            use_auth_token: str,
            device: Optional[str] = None,
            chunk_length: Optional[float] = None,
            overlap_length: Optional[float] = None,
            min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None
            ) -> Tuple[List[Segment], float]:
        """
        Diarize transcribed result as a post-processing

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio input. This can be file path or binary type.
        transcribed_result: List[Segment]
            transcribed result through whisper.
        use_auth_token: str
            Huggingface token with READ permission. This is only needed the first time you download the model.
            You must manually go to the website https://huggingface.co/pyannote/speaker-diarization-community-1 and agree to their TOS to download the model.
        device: Optional[str]
            Device for diarization.

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for running
        """
        log_vram("DIARIZER.run:start", f"transcribed_segments={len(transcribed_result)}")
        run_start_stats = get_vram_stats()
        start_time = time.time()

        if device is None:
            device = self.device

        if device != self.device or self.pipe is None:
            log_vram("DIARIZER.run:loading_model", f"device={device}")
            self.update_pipe(
                device=device,
                use_auth_token=use_auth_token,
                model_name="pyannote/speaker-diarization-community-1",
                chunk_length=chunk_length,
                overlap_length=overlap_length
            )
            log_vram_delta("DIARIZER.run:model_loaded", run_start_stats)
        elif chunk_length is not None or overlap_length is not None:
            # Update pipeline configuration if parameters changed
            if chunk_length is not None:
                self.pipe.model._segmentation.duration = chunk_length
            if overlap_length is not None and chunk_length is not None:
                step = chunk_length - overlap_length
                self.pipe.model._segmentation.step = step

        audio = load_audio(audio)
        log_vram("DIARIZER.run:audio_loaded", f"audio_shape={audio.shape if hasattr(audio, 'shape') else 'N/A'}")

        pre_pipe_stats = get_vram_stats()
        log_vram("DIARIZER.run:before_pipe_call", "Calling diarization pipeline")
        
        diarization_segments = self.pipe(audio, min_speakers=min_speakers, max_speakers=max_speakers)
        
        log_vram_delta("DIARIZER.run:after_pipe_call", pre_pipe_stats, f"segments_count={len(diarization_segments) if hasattr(diarization_segments, '__len__') else 'DataFrame'}")

        log_vram("DIARIZER.run:before_assign_speakers", "Assigning speakers to words")
        diarized_result = assign_word_speakers(
            diarization_segments,
            {"segments": transcribed_result}
        )
        log_vram("DIARIZER.run:after_assign_speakers", f"result_segments={len(diarized_result['segments'])}")

        segments_result = []
        for segment in diarized_result["segments"]:
            speaker = "None"
            if "speaker" in segment:
                speaker = segment["speaker"]
            diarized_text = speaker + "|" + segment["text"].strip()
            segments_result.append(Segment(
                start=segment["start"],
                end=segment["end"],
                text=diarized_text
            ))

        log_vram("DIARIZER.run:before_cleanup", f"result_segments={len(segments_result)}")
        pre_cleanup_stats = get_vram_stats()

        # Aggressively clean up intermediate data structures
        del audio
        del diarization_segments
        del diarized_result
        
        # Force GPU cache cleanup after diarization to prevent memory surge
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations complete before proceeding
            torch.cuda.reset_max_memory_allocated()  # Reset peak memory tracking
        
        # Force garbage collection to free Python objects
        gc.collect()
        
        log_vram_delta("DIARIZER.run:after_cleanup", pre_cleanup_stats, "Cleanup complete")
        log_vram_delta("DIARIZER.run:end", run_start_stats, f"Total time: {time.time() - start_time:.2f}s")
        
        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_pipe(self,
                    use_auth_token: Optional[str] = None,
                    device: Optional[str] = None,
                    model_name: Optional[str] = None,
                    chunk_length: Optional[float] = None,
                    overlap_length: Optional[float] = None,
                    ):
        """
        Set pipeline for diarization

        Parameters
        ----------
        use_auth_token: str
            Huggingface token with READ permission. This is only needed the first time you download the model.
            You must manually go to the website https://huggingface.co/pyannote/speaker-diarization-community-1 and agree to their TOS to download the model.
        device: str
            Device for diarization.
        model_name: str
            Model name for diarization. Defaults to pyannote/speaker-diarization-community-1
        """
        if device is None:
            device = self.get_device()
        self.device = device

        if model_name is None:
            model_name = "pyannote/speaker-diarization-community-1"

        os.makedirs(self.model_dir, exist_ok=True)

        if (not os.listdir(self.model_dir) and
                not use_auth_token):
            print(
                "\nFailed to diarize. You need huggingface token and agree to their requirements to download the diarization model.\n"
                "Go to \"https://huggingface.co/pyannote/speaker-diarization-community-1\" and follow their instructions to download the model.\n"
            )
            return

        logger = logging.getLogger("speechbrain.utils.train_logger")
        # Disable redundant torchvision warning message
        logger.disabled = True
        self.pipe = DiarizationPipeline(
            model_name=model_name,
            token=use_auth_token,  # Use 'token' parameter for pyannote.audio 4.x
            device=device,
            cache_dir=self.model_dir,
            chunk_length=chunk_length,
            overlap_length=overlap_length
        )
        logger.disabled = False

    def offload(self):
        """Offload the model and free up the memory"""
        log_vram("DIARIZER.offload:start", "Beginning model offload")
        pre_offload_stats = get_vram_stats()
        
        if self.pipe is not None:
            # Move model to CPU before deletion to ensure proper cleanup
            if hasattr(self.pipe, 'model') and self.pipe.model is not None:
                try:
                    log_vram("DIARIZER.offload:moving_to_cpu", "Moving model to CPU")
                    # Ensure model is in eval mode before moving
                    if hasattr(self.pipe.model, 'eval'):
                        self.pipe.model.eval()
                    # Move model to CPU
                    self.pipe.model = self.pipe.model.cpu()
                    log_vram("DIARIZER.offload:moved_to_cpu", "Model moved to CPU")
                    # Clear any cached computations
                    if hasattr(self.pipe.model, 'zero_grad'):
                        self.pipe.model.zero_grad()
                except Exception as e:
                    log_vram("DIARIZER.offload:move_failed", f"Error: {e}")
                    pass  # If moving fails, continue with deletion
            del self.pipe
            self.pipe = None
            log_vram("DIARIZER.offload:pipe_deleted", "Pipeline deleted")
        if self.device == "cuda":
            # Clear GPU cache multiple times to ensure all memory is freed
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations complete
            torch.cuda.empty_cache()  # Second pass to catch any delayed releases
            torch.cuda.reset_max_memory_allocated()
        if self.device == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_accumulated_memory_stats()
            torch.xpu.reset_peak_memory_stats()
        # Force garbage collection to free Python objects
        gc.collect()
        
        log_vram_delta("DIARIZER.offload:end", pre_offload_stats, "Offload complete")

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def get_available_device():
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.xpu.is_available():
            devices.append("xpu")
        if torch.backends.mps.is_available():
            devices.append("mps")
        return devices