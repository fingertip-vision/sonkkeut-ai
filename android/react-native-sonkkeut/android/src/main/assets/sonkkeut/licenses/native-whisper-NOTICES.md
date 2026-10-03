# Native ASR dependencies

The source bundle preserves each project's complete license and copyright notices.

- OpenNMT CTranslate2 4.8.2: MIT, `CTranslate2/LICENSE`.
- Google Ruy: Apache-2.0, `CTranslate2/third_party/ruy/LICENSE`.
- Google cpu_features: Apache-2.0, `CTranslate2/third_party/cpu_features/LICENSE`.
- PyTorch cpuinfo: BSD-2-Clause, `CTranslate2/third_party/ruy/third_party/cpuinfo/LICENSE`.
- spdlog: MIT, `CTranslate2/third_party/spdlog/LICENSE`; bundled fmt notices remain in its source.
- GoogleTest, present as the pinned Ruy dependency but not linked in this build: BSD-3-Clause, `CTranslate2/third_party/ruy/third_party/googletest/LICENSE`.
- nlohmann JSON, half_float, BS_thread_pool and math helper headers: their original notices remain under `CTranslate2/third_party`.
- OpenAI Whisper's exact mel filterbank and tokenizer byte-alphabet/BPE contract: MIT, https://github.com/openai/whisper/blob/main/LICENSE.
- Android NDK libc++ runtime: LLVM Apache-2.0 with LLVM exception. This toolchain runtime is packaged separately from the source-only bundle; retain its notices when distributing binaries.

The user-provided Whisper weights are downloaded separately from the existing fingertip-vision release. This bundle does not redistribute or alter those weights.
