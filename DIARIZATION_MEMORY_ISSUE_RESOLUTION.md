# Diarization Memory Issue Resolution

## Executive Summary

During implementation of `pyannote.audio 4.x` with the `pyannote/speaker-diarization-community-1` model, we discovered a critical memory surge issue affecting specific chunk size configurations. This document details the problem, investigation process, root cause analysis, and the final solution.

## Problem Description

When processing audio files with speaker diarization enabled, we observed excessive VRAM usage (9+ GB) for specific chunk size configurations, while other configurations used normal memory levels (2-3 GB). The memory surge occurred specifically when using chunk lengths between 6-10 seconds, with peak memory usage reaching 9-10 GB compared to the expected 2-3 GB.

### Symptoms

- **Memory Surge**: VRAM usage spiked to 9+ GB during diarization inference
- **Specific Range**: Only chunk sizes 6-10 seconds triggered the surge
- **Normal Behavior**: Chunks < 5s or > 11s used normal memory (2-3 GB)
- **No Leak**: Memory was properly released after processing, indicating allocation rather than leak

## Investigation Process

### Initial Hypothesis

We initially suspected:
1. Memory leaks in our cleanup code
2. Inefficient tensor management
3. PyTorch allocator behavior
4. Model loading/unloading issues

### Approaches Attempted

#### 1. Enhanced Memory Cleanup
**Hypothesis**: Insufficient cleanup of GPU tensors and intermediate data structures.

**Actions Taken**:
- Added explicit tensor deletion (`del audio_tensor`)
- Implemented `torch.cuda.empty_cache()` and `torch.cuda.synchronize()` calls
- Moved models to CPU before deletion
- Added garbage collection (`gc.collect()`)
- Replaced `deepcopy` with shallow copies for audio data

**Outcome**: No improvement. Memory surge persisted, confirming the issue was not in our cleanup logic.

#### 2. VRAM Diagnostic Logging
**Hypothesis**: Need granular visibility into memory usage at each processing stage.

**Actions Taken**:
- Created comprehensive VRAM diagnostic logging system
- Added checkpoints throughout the pipeline (transcription, diarization, file generation)
- Tracked allocated, reserved, and peak memory at each stage

**Outcome**: Confirmed that the surge occurred during diarization inference itself, not during cleanup or other stages. Peak memory was primarily "Reserved" by PyTorch's allocator, not "Allocated".

#### 3. Model Architecture Analysis
**Hypothesis**: Internal model architecture might have hard-coded window sizes or thresholds.

**Actions Taken**:
- Inspected `pyannote.audio.Pipeline` internal structure
- Examined `Inference` class parameters and defaults
- Checked for magic numbers or constants in the codebase
- Analyzed LSTM configuration (input size: 60, hidden size: 128, 4 layers, bidirectional)

**Outcome**: No hard-coded values found that would explain the threshold behavior. The model architecture itself did not reveal the cause.

#### 4. Parameter Tuning Research
**Hypothesis**: Library documentation or community knowledge might reveal optimal configurations.

**Actions Taken**:
- Researched `pyannote.audio` documentation
- Searched GitHub issues and discussions
- Reviewed community recommendations for memory optimization
- Tested various `step` and `batch_size` configurations

**Outcome**: Found that adjusting `step` (overlap) could improve performance, but did not resolve the memory surge for the problematic chunk size range.

### Breakthrough: Systematic Testing

We created diagnostic scripts to systematically test different chunk size configurations:

**Test Results**:
- **4s chunks**: 2.78 GB peak memory
- **6s chunks**: 9.38 GB peak memory ⚠️
- **8s chunks**: 9.20 GB peak memory ⚠️
- **10s chunks**: 9.99 GB peak memory ⚠️
- **12s chunks**: 1.98 GB peak memory

**Key Finding**: The memory surge was NOT correlated with:
- Number of chunks (500 chunks used less memory than 333 chunks)
- Batch count (similar batch counts showed different memory)
- Chunk size in MB (larger chunks used less memory)

This confirmed that the issue was internal to `pyannote.audio`, not our code or data characteristics.

## Root Cause Analysis

### Internal Threshold in pyannote.audio

The investigation revealed that `pyannote.audio` has an internal optimization or buffer allocation strategy that triggers different memory allocation behavior for chunk sizes in the 6-10 second range. This appears to be:

1. **An internal threshold**: The library switches processing strategies around 5-6 seconds
2. **Optimization that backfires**: An optimization intended to improve performance for certain chunk sizes actually causes excessive memory allocation
3. **Not a bug in our code**: Our parameter modifications work correctly; the issue is in the library's internal handling

### Evidence

- Duration modification works correctly (verified at all tested values)
- Memory surge occurs specifically for 6-10s chunks regardless of other factors
- Pattern is consistent across different audio files
- Memory is properly released after processing (no leak)
- Issue persists even with optimal batch sizes and cleanup

## Solution

### Optimal Configuration

Based on testing, the optimal configuration is:
- **Chunk Length**: 12 seconds
- **Overlap Length**: 6 seconds (50% overlap)
- **Result**: ~2 GB peak memory, fast processing (~7.6s for 17-minute audio)

### Implementation

We implemented the following changes:

1. **Added User-Configurable Parameters**:
   - `chunk_length`: Configurable chunk size (default: 12.0s)
   - `overlap_length`: Configurable overlap (default: 6.0s)
   - `min_speakers`: Minimum speakers to detect (default: 1)
   - `max_speakers`: Maximum speakers to detect (default: 10)

2. **UI Warning**: Added prominent warning in the user interface advising users to avoid chunk sizes between 6-10 seconds.

3. **Runtime Validation**: Added validation and warnings when users attempt to use problematic chunk sizes.

4. **Default Configuration**: Set optimal defaults (12s chunk, 6s overlap) in `default_parameters.yaml`.

### Code Changes

- **`modules/whisper/data_classes.py`**: Added new fields to `DiarizationParams`
- **`modules/diarize/diarize_pipeline.py`**: Added parameter configuration and validation
- **`modules/diarize/diarizer.py`**: Updated to pass new parameters through the pipeline
- **`modules/whisper/base_transcription_pipeline.py`**: Updated to extract and pass diarization parameters
- **`configs/default_parameters.yaml`**: Added optimal default values

## Recommendations

### For Users

1. **Use Recommended Values**: Stick with the default 12s chunk / 6s overlap configuration for optimal memory usage
2. **Avoid Problematic Range**: Do not use chunk sizes between 6-10 seconds
3. **Alternative Options**:
   - **Small chunks** (< 5s): Lower memory but slower processing
   - **Large chunks** (> 11s): Lower memory and faster processing

### For Developers

1. **Monitor Memory**: If implementing custom chunk sizes, monitor VRAM usage
2. **Test Thoroughly**: Test memory usage with different chunk sizes before deploying
3. **Document Limitations**: Document any known limitations or workarounds

## Lessons Learned

1. **Not All Memory Issues Are Leaks**: High memory usage doesn't always indicate a leak; it can be inefficient allocation strategies
2. **Library Internals Matter**: Understanding library internals is crucial for diagnosing performance issues
3. **Systematic Testing**: Systematic testing with controlled variables is essential for identifying root causes
4. **Documentation of Failed Approaches**: Documenting failed approaches helps prevent repeating unsuccessful experiments

## Future Considerations

1. **Monitor pyannote.audio Updates**: Future versions may fix this internal threshold issue
2. **Alternative Models**: Consider testing alternative diarization models if memory constraints are critical
3. **Chunking Strategy**: For very long audio files, consider implementing custom chunking logic outside the library

## Known Warnings

During normal operation, the following warnings may appear in logs. These are informational and do not affect functionality:

### 1. TF32 Disabled Warning
**Source**: `pyannote/audio/utils/reproducibility.py`  
**Message**: TensorFloat-32 (TF32) has been disabled for reproducibility  
**Impact**: None - This is intentional behavior by pyannote.audio  
**Action**: No action needed

### 2. Statistical Pooling Warning
**Source**: `pyannote/audio/models/blocks/pooling.py`  
**Message**: `std(): degrees of freedom is <= 0`  
**Impact**: Minor - Occurs when processing very short segments or edge cases  
**Action**: Monitor for accuracy issues, but typically harmless

### 3. FutureWarning: reset_max_memory_allocated
**Source**: PyTorch CUDA memory API  
**Message**: API evolution warning about `torch.cuda.reset_max_memory_allocated`  
**Impact**: None - Functionality unchanged  
**Action**: Can be addressed in future PyTorch version updates

## References

- `pyannote.audio` Documentation: https://github.com/pyannote/pyannote-audio
- Model Card: https://huggingface.co/pyannote/speaker-diarization-community-1
- Diagnostic Scripts: Located in `../diagnostics/` directory
- Performance Summary: See `PERFORMANCE_SUMMARY.md` for detailed timing and memory analysis

## Appendix: Failed Approaches Summary

### Approach 1: Enhanced Memory Cleanup
- **Hypothesis**: Insufficient cleanup causing memory accumulation
- **Actions**: Added explicit tensor deletion, cache clearing, CPU offloading
- **Result**: No improvement - confirmed issue was in allocation, not cleanup

### Approach 2: Diagnostic Logging
- **Hypothesis**: Need visibility into memory usage patterns
- **Actions**: Implemented comprehensive VRAM tracking throughout pipeline
- **Result**: Identified that surge occurs during inference, not cleanup

### Approach 3: Model Architecture Analysis
- **Hypothesis**: Hard-coded values or thresholds in model architecture
- **Actions**: Inspected internal structure, checked for magic numbers
- **Result**: No hard-coded values found that would explain behavior

### Approach 4: Parameter Tuning
- **Hypothesis**: Optimal parameters might reduce memory usage
- **Actions**: Tested various step, batch_size, and duration combinations
- **Result**: Found performance improvements but did not resolve memory surge for problematic range

### Approach 5: Systematic Testing
- **Hypothesis**: Need to understand exact conditions triggering memory surge
- **Actions**: Created diagnostic scripts to test different chunk size configurations
- **Result**: Identified specific problematic range (6-10s) and confirmed it's internal to library

---

**Document Version**: 1.0  
**Date**: 2025-01-05  
**Status**: Resolved
