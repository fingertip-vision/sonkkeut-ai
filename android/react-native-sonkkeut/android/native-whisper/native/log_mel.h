#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
// Input: mono PCM16 at exactly 16 kHz, 1..480000 samples.
// Output: row-major float32 [80, 3000], matching Whisper's 30-second pad/trim.
std::vector<float> whisper_log_mel(const int16_t* pcm, size_t samples);
