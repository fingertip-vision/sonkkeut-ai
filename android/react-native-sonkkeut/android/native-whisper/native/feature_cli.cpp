#include "log_mel.h"
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argc, char** argv) {
  if (argc != 3) { std::cerr << "Usage: feature_cli PCM16LE.raw logmel.f32\n"; return 2; }
  try {
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) return 3;
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
    if (bytes.size() % 2 || bytes.empty() || bytes.size() > 960000) return 4;
    std::vector<int16_t> pcm(bytes.size() / 2);
    for (size_t i = 0; i < pcm.size(); ++i)
      pcm[i] = static_cast<int16_t>(static_cast<unsigned char>(bytes[2 * i]) | (static_cast<unsigned char>(bytes[2 * i + 1]) << 8));
    auto mel = whisper_log_mel(pcm.data(), pcm.size());
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(mel.data()), mel.size() * sizeof(float));
    return out ? 0 : 5;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 6; }
}
