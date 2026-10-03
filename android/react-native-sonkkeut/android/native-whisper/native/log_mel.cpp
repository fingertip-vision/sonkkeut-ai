#include "log_mel.h"
#include "mel_filters_80.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace {
constexpr int kFft = 400, kHop = 160, kFrames = 3000, kSamples = 480000;
using Complex = std::complex<float>;
struct FftPlan {
  std::array<Complex, kFft> twiddle;
  std::array<float, kFft> window;
  FftPlan() {
    constexpr double pi = 3.14159265358979323846;
    for (int i = 0; i < kFft; ++i) {
      const double angle = -2 * pi * i / kFft;
      twiddle[i] = Complex(static_cast<float>(std::cos(angle)), static_cast<float>(std::sin(angle)));
      window[i] = static_cast<float>(0.5 - 0.5 * std::cos(2 * pi * i / kFft));
    }
  }
  // Mixed-radix Cooley-Tukey FFT for 400 = 2^4 * 5^2. No FFT padding.
  void transform(const Complex* input, int stride, int n, Complex* output) const {
    if (n == 1) { output[0] = input[0]; return; }
    const int radix = n % 2 == 0 ? 2 : 5;
    const int m = n / radix;
    for (int j = 0; j < radix; ++j)
      transform(input + j * stride, stride * radix, m, output + j * m);
    std::array<Complex, kFft> combined;
    for (int l = 0; l < radix; ++l) {
      for (int k = 0; k < m; ++k) {
        Complex sum(0, 0);
        const int index = k + m * l;
        for (int j = 0; j < radix; ++j)
          sum += output[j * m + k] * twiddle[(j * index * (kFft / n)) % kFft];
        combined[index] = sum;
      }
    }
    std::copy_n(combined.begin(), n, output);
  }
};
}

std::vector<float> whisper_log_mel(const int16_t* pcm, size_t samples) {
  if (!pcm || samples == 0 || samples > kSamples)
    throw std::invalid_argument("Whisper requires 1..480000 mono PCM16 samples at 16 kHz");
  static const FftPlan plan;
  std::vector<float> audio(kSamples, 0.0f);
  for (size_t i = 0; i < samples; ++i) audio[i] = pcm[i] / 32768.0f;
  std::vector<float> mel(80 * kFrames);
  std::array<Complex, kFft> input, spectrum;
  std::array<float, 201> power;
  float peak = -10.0f;
  for (int frame = 0; frame < kFrames; ++frame) {
    for (int i = 0; i < kFft; ++i) {
      int index = frame * kHop + i - kFft / 2;
      // torch.stft(center=True, pad_mode="reflect") after 30-second pad/trim.
      if (index < 0) index = -index;
      if (index >= kSamples) index = 2 * kSamples - 2 - index;
      input[i] = Complex(audio[index] * plan.window[i], 0);
    }
    plan.transform(input.data(), 1, kFft, spectrum.data());
    for (int k = 0; k <= kFft / 2; ++k) power[k] = std::norm(spectrum[k]);
    for (int band = 0; band < 80; ++band) {
      float energy = 0;
      for (int k = 0; k <= kFft / 2; ++k) energy += kWhisperMel80[band][k] * power[k];
      const float value = std::log10(std::max(energy, 1e-10f));
      mel[band * kFrames + frame] = value;
      peak = std::max(peak, value);
    }
  }
  for (float& value : mel) value = (std::max(value, peak - 8.0f) + 4.0f) / 4.0f;
  return mel;
}
