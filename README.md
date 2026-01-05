# Whisper-WebUI (PyAnnote 4.x Fork)

A Gradio-based browser interface for [Whisper](https://github.com/openai/whisper) with upgraded speaker diarization support. This fork extends the original [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI) with `pyannote.audio 4.x` and the `pyannote/speaker-diarization-community-1` model, along with significant memory optimizations and configurable diarization parameters.

![screen](https://github.com/user-attachments/assets/caea3afd-a73c-40af-a347-8d57914b1d0f)

## 🙏 Credits and Attribution

**This fork is built upon the excellent work of the original Whisper-WebUI project.**

### Original Project
- **Repository**: [jhj0517/Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI)
- **Maintainer**: [jhj0517](https://github.com/jhj0517)
- **License**: See [LICENSE](LICENSE) file

The original Whisper-WebUI is a well-developed, feature-rich project that includes:
- Multiple Whisper implementation support (openai/whisper, faster-whisper, insanely-fast-whisper)
- Subtitle generation from files, YouTube, and microphone
- Speech-to-text and text-to-text translation (NLLB, DeepL)
- Voice Activity Detection (Silero VAD)
- Background music separation (UVR)
- REST API backend
- Comprehensive Gradio-based UI

**All of these features were implemented by the original authors and contributors.** This fork focuses specifically on upgrading the speaker diarization component and optimizing memory usage.

### This Fork's Contributions

This fork adds the following improvements:
- **Speaker diarization upgrade**: `pyannote.audio 4.x` with `speaker-diarization-community-1` model
- **Memory optimizations**: VRAM surge resolution and enhanced GPU memory management
- **Configurable parameters**: User-adjustable diarization settings
- **Docker improvements**: Enhanced development workflow and platform-specific support
- **Diagnostic tooling**: Optional VRAM diagnostics module

See the ["What's Different"](#-whats-different-in-this-fork) section below for detailed information about these changes.

## 🆕 What's Different in This Fork?

This fork introduces several architectural and operational improvements focused on speaker diarization:

### **Speaker Diarization Upgrade**
- **Upgraded to `pyannote.audio 4.x`** with the `pyannote/speaker-diarization-community-1` model (replacing the deprecated `speaker-diarization-3.1`)
- **Platform-specific installation requirements** (see Installation section below)
- **Configurable diarization parameters** for optimal performance:
  - Chunk length and overlap control
  - Minimum/maximum speaker limits
  - Memory-optimized defaults (12s chunk / 6s overlap)

### **Memory Optimizations**
- **Resolved VRAM surge issue**: Fixed excessive memory usage (9+ GB → ~2 GB peak) for specific chunk configurations
- **Enhanced GPU memory management**: Improved cleanup across all models (Whisper, Diarization, VAD, NLLB, UVR)
- **Optimal default configuration**: Pre-configured settings that avoid problematic memory patterns

### **Docker Development Workflow**
- **Rapid iteration support**: Live code editing without full image rebuilds
- **Ubuntu 24.04 LTS base**: Updated to latest LTS with Python 3.12
- **Improved venv persistence**: Named volumes for reliable environment management

### **Diagnostic Tooling**
- **VRAM diagnostics module**: Optional detailed memory tracking (disabled by default)
- **Comprehensive logging**: Granular checkpoint logging throughout the pipeline

## ⚠️ Important: Platform-Specific Installation Requirements

**Speaker diarization with `pyannote.audio 4.x` has different installation requirements depending on your operating system:**

### **Windows Users** 🪟
**Speaker diarization is ONLY available via Docker.**

`pyannote.audio 4.x` requires `torchcodec`, which has **no pip-compatible wheels for Windows**. The only viable solution is to use Docker, which provides a Linux environment where all dependencies can be installed correctly.

- ✅ **Use Docker** (recommended for Windows users)
- ❌ **Local installation will fail** for speaker diarization features

### **Linux Users** 🐧
**Both Docker and local installation are supported.**

Linux users can install `pyannote.audio 4.x` directly via pip, as `torchcodec` has pip wheels available for Linux.

- ✅ **Use Docker** (recommended for consistency)
- ✅ **Use local installation** (via `Install.sh`)

## Notebook
If you wish to try this on Colab, you can use the original project's notebook [here](https://colab.research.google.com/github/jhj0517/Whisper-WebUI/blob/master/notebook/whisper-webui.ipynb)!

# Features

**Note**: The following features were implemented by the original Whisper-WebUI project. This fork maintains all original functionality while upgrading the speaker diarization component.

- Select the Whisper implementation you want to use between :
   - [openai/whisper](https://github.com/openai/whisper)
   - [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) (used by default)
   - [Vaibhavs10/insanely-fast-whisper](https://github.com/Vaibhavs10/insanely-fast-whisper)
- Generate subtitles from various sources, including :
  - Files
  - Youtube
  - Microphone
- Currently supported subtitle formats : 
  - SRT
  - WebVTT
  - txt ( only text file without timeline )
- Speech to Text Translation 
  - From other languages to English. ( This is Whisper's end-to-end speech-to-text translation feature )
- Text to Text Translation
  - Translate subtitle files using Facebook NLLB models
  - Translate subtitle files using DeepL API
- Pre-processing audio input with [Silero VAD](https://github.com/snakers4/silero-vad).
- Pre-processing audio input to separate BGM with [UVR](https://github.com/Anjok07/ultimatevocalremovergui). 
- Post-processing with speaker diarization using the [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) model.
   - **🆕 New in this fork**: Upgraded from deprecated `speaker-diarization-3.1` to `speaker-diarization-community-1`
   - **🆕 New in this fork**: Configurable chunk length, overlap, and speaker limits for optimal performance
   - **🆕 New in this fork**: Memory-optimized default configuration avoids problematic memory patterns
   - To download the pyannote model, you need to have a Huggingface token and manually accept their terms in the pages below.
      1. https://huggingface.co/pyannote/speaker-diarization-community-1
      2. https://huggingface.co/pyannote/segmentation-3.0

### Pipeline Diagram
![Transcription Pipeline](https://github.com/user-attachments/assets/1d8c63ac-72a4-4a0b-9db0-e03695dcf088)

# Installation and Running

## 🐳 Running with Docker (Recommended)

Docker is the **recommended and only supported method for Windows users** who want speaker diarization. Linux users can also use Docker for consistency and easier dependency management.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- NVIDIA GPU with CUDA support (for GPU acceleration)
- Docker Desktop configured with WSL2 backend (Windows)

### Steps

1. **Clone this repository**
```sh
git clone https://github.com/gabriel-filincowsky/Whisper-WebUI-Pyannote4.git
cd Whisper-WebUI-Pyannote4
```

2. **Build the Docker image** (Image is about 7GB~)
```sh
docker compose build
```

3. **Run the container**
```sh
docker compose up -d
```

4. **Access the WebUI**
Open your browser at `http://localhost:7860`

### Docker Configuration

The `docker-compose.yaml` file is pre-configured for:
- GPU access via NVIDIA Container Toolkit
- Volume mounts for models, outputs, and configs (persisted on host)
- Live code editing (source code mounted for development)
- Automatic venv restoration

**🆕 New in this fork**: Enhanced Docker workflow with rapid development iteration support.

If needed, update the [`docker-compose.yaml`](docker-compose.yaml) to match your environment (e.g., remove GPU section if using CPU).

### Development Workflow

**🆕 New in this fork**: This fork supports rapid development iteration:
- Source code is mounted as a volume, so code changes reflect immediately
- Restart the container with `docker compose restart` to apply changes
- No need to rebuild the image for code modifications

## 💻 Running Locally (Linux Only)

**⚠️ Windows users**: Local installation will **not work** for speaker diarization features due to `torchcodec` dependency limitations. Use Docker instead.

### Prerequisites
To run this WebUI locally, you need:
- `git`
- `3.10 <= python <= 3.12`
- `FFmpeg`
- **Linux operating system** (Windows users must use Docker)

**Edit `--extra-index-url` in the [`requirements.txt`](requirements.txt) to match your device.**  
By default, the WebUI assumes you're using an Nvidia GPU and **CUDA 12.8.** If you're using Intel or another CUDA version, read the [`requirements.txt`](requirements.txt) and edit `--extra-index-url`.

Please follow the links below to install the necessary software:
- git : [https://git-scm.com/downloads](https://git-scm.com/downloads)
- python : [https://www.python.org/downloads/](https://www.python.org/downloads/) **`3.10 ~ 3.12` is recommended.** 
- FFmpeg :  [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- CUDA : [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

After installing FFmpeg, **make sure to add the `FFmpeg/bin` folder to your system PATH!**

### Installation Using the Script Files

1. **Clone this repository**
```shell
git clone https://github.com/gabriel-filincowsky/Whisper-WebUI-Pyannote4.git
cd Whisper-WebUI-Pyannote4
```

2. **Run `install.sh` to install dependencies**
   - This will create a `venv` directory and install dependencies there
   - On Linux, `pyannote.audio 4.x` and `torchcodec` will install correctly via pip

3. **Start WebUI with `start-webui.sh`**
   - This will run `python app.py` after activating the venv

You can also run the project with command line arguments if you like to, see the [original project's wiki](https://github.com/jhj0517/Whisper-WebUI/wiki/Command-Line-Arguments) for a guide to arguments.

## 📊 Configurable Diarization Parameters

**🆕 New in this fork**: This fork exposes several diarization parameters for optimal performance:

- **Chunk Length** (default: 12.0 seconds)
  - Controls the size of audio chunks processed by the diarization model
  - **⚠️ Warning**: Avoid values between 6-10 seconds as they trigger excessive memory usage (9+ GB)
  - Recommended: 12 seconds or less than 5 seconds

- **Overlap Length** (default: 6.0 seconds)
  - Overlap between consecutive chunks
  - Should be less than chunk length
  - Recommended: 50% of chunk length

- **Minimum Speakers** (default: 1)
  - Minimum number of speakers to detect

- **Maximum Speakers** (default: 4)
  - Maximum number of speakers to detect
  - Lower values can improve performance and accuracy

These parameters are available in the WebUI's diarization settings panel.

## 🔧 VRAM Usage and Performance

This project is integrated with [faster-whisper](https://github.com/guillaumekln/faster-whisper) by default for better VRAM usage and transcription speed. This integration was implemented by the original Whisper-WebUI project.

According to faster-whisper, the efficiency of the optimized whisper model is as follows: 
| Implementation    | Precision | Beam size | Time  | Max. GPU memory | Max. CPU memory |
|-------------------|-----------|-----------|-------|-----------------|-----------------|
| openai/whisper    | fp16      | 5         | 4m30s | 11325MB         | 9439MB          |
| faster-whisper    | fp16      | 5         | 54s   | 4755MB          | 3244MB          |

### Memory Optimizations in This Fork

**🆕 New in this fork**: Additional memory optimizations beyond the original faster-whisper integration:

- **Diarization VRAM**: Reduced from 9+ GB to ~2 GB peak usage with optimal configuration
- **Enhanced cleanup**: Improved GPU memory management across all models
- **Optimal defaults**: Pre-configured settings that avoid memory surge patterns

If you want to use an implementation other than faster-whisper, use `--whisper_type` arg and the repository name.  
Read the [original project's wiki](https://github.com/jhj0517/Whisper-WebUI/wiki/Command-Line-Arguments) for more info about CLI args.

If you want to use a fine-tuned model, manually place the models in `models/Whisper/` corresponding to the implementation.

Alternatively, if you enter the huggingface repo id (e.g, [deepdml/faster-whisper-large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)) in the "Model" dropdown, it will be automatically downloaded in the directory.

![image](https://github.com/user-attachments/assets/76487a46-b0a5-4154-b735-ded73b2d83d4)

# REST API
If you're interested in deploying this app as a REST API, please check out [/backend](backend). The REST API backend was implemented by the original Whisper-WebUI project.

## 📚 Additional Documentation

**🆕 New in this fork**: For detailed information about the memory optimizations and technical implementation:

- **[Diarization Memory Issue Resolution](DIARIZATION_MEMORY_ISSUE_RESOLUTION.md)**: Comprehensive documentation of the VRAM surge investigation and resolution
- **[Pull Request Strategy](PULL_REQUEST_STRATEGY.md)**: Guidance on contributing improvements upstream

## 🤝 Contributing and Pull Requests

This fork maintains compatibility with the upstream repository structure while introducing significant architectural changes. Key differences include:

- **Docker-first approach** for Windows users (due to `torchcodec` limitations)
- **Memory optimizations** and configurable parameters
- **Updated dependencies** (`pyannote.audio 4.x`, PyTorch 2.8.0+)

**Note on Upstream Contribution**: While this fork is fully functional and well-documented, some changes (particularly the Docker-first Windows workflow) may be too disruptive for direct upstream integration. However, core improvements like:
- Memory optimizations
- Configurable diarization parameters
- API compatibility fixes (`itertracks` → `speaker_diarization`)
- Enhanced GPU memory management

...could potentially be contributed upstream as separate, focused pull requests.

See [PULL_REQUEST_STRATEGY.md](PULL_REQUEST_STRATEGY.md) for detailed guidance on upstream contribution strategy.

## 🗓 TODO (This Fork)

The following items are specific to this fork's development:

- [x] Upgrade to pyannote.audio 4.x with speaker-diarization-community-1 model
- [x] Resolve VRAM surge issues with optimal chunk configuration
- [x] Add configurable diarization parameters
- [x] Enhance Docker development workflow
- [x] Add VRAM diagnostic tooling
- [ ] Improve Windows installation documentation
- [ ] Add performance benchmarking suite

**Note**: The original project has its own TODO list. See the [original repository](https://github.com/jhj0517/Whisper-WebUI) for features planned by the upstream maintainers.

### Translation 🌐
Any PRs that translate the language into [translation.yaml](configs/translation.yaml) would be greatly appreciated! This feature was implemented by the original Whisper-WebUI project.
