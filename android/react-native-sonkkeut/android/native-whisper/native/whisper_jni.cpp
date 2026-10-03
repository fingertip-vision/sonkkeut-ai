#include <jni.h>
#include <ctranslate2/models/whisper.h>
#include <nlohmann/json.hpp>
#include "log_mel.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace {
using Whisper = ctranslate2::models::Whisper;
std::mutex registry_mutex;
std::unordered_map<jlong, std::shared_ptr<Whisper>> registry;
std::atomic<jlong> next_id{1};
void fail(JNIEnv* env, const std::exception& error) {
  env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), error.what());
}
std::shared_ptr<Whisper> get_model(jlong handle) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto it = registry.find(handle);
  if (it == registry.end()) throw std::invalid_argument("Whisper model is closed or not initialized");
  return it->second;
}
jstring generate(JNIEnv* env, const std::shared_ptr<Whisper>& model, const std::vector<float>& mel,
                 const std::vector<std::string>& hints = {}) {
  using namespace ctranslate2;
  const auto begin = std::chrono::steady_clock::now();
  StorageView features({1, 80, 3000}, mel, Device::CPU);
  models::WhisperOptions options;
  options.beam_size = 1;
  options.max_length = 448;
  options.return_scores = true;
  options.return_no_speech_prob = true;
  std::vector<std::string> tokens;
  if (!hints.empty()) { tokens.emplace_back("<|startofprev|>"); tokens.insert(tokens.end(), hints.begin(), hints.end()); }
  tokens.insert(tokens.end(), {"<|startoftranscript|>", "<|ko|>", "<|transcribe|>", "<|notimestamps|>"});
  const std::vector<std::vector<std::string>> prompt = {tokens};
  const auto result = model->generate(features, prompt, options).at(0).get();
  nlohmann::json data = {
    {"token_ids", result.sequences_ids.at(0)},
    {"no_speech_probability", result.no_speech_prob},
    {"inference_ms", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count()}
  };
  // CT2's length-normalized sequence score is not a calibrated word confidence.
  if (!result.scores.empty()) data["sequence_score"] = result.scores[0];
  return env->NewStringUTF(data.dump(-1, ' ', true).c_str());
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_kr_sonkkeut_android_KoreanWhisperNative_create(JNIEnv* env, jobject, jstring path, jint threads, jboolean quantized) {
  try {
    if (!path || threads < 1 || threads > 4) throw std::invalid_argument("Invalid model path or thread count");
    const char* value = env->GetStringUTFChars(path, nullptr);
    if (!value) return 0;
    std::string model_path(value);
    env->ReleaseStringUTFChars(path, value);
    ctranslate2::ReplicaPoolConfig config;
    config.num_threads_per_replica = threads;
    config.max_queued_batches = 1;
    auto model = std::make_shared<Whisper>(model_path, ctranslate2::Device::CPU,
      quantized ? ctranslate2::ComputeType::INT8_FLOAT32 : ctranslate2::ComputeType::FLOAT32,
      std::vector<int>{0}, false, config);
    if (model->n_mels() != 80 || !model->is_multilingual())
      throw std::invalid_argument("This adapter requires the multilingual 80-mel Whisper model");
    const auto handle = next_id.fetch_add(1);
    std::lock_guard<std::mutex> lock(registry_mutex);
    registry.emplace(handle, std::move(model));
    return handle;
  } catch (const std::exception& e) { fail(env, e); return 0; }
}

extern "C" JNIEXPORT jstring JNICALL
Java_kr_sonkkeut_android_KoreanWhisperNative_transcribe(JNIEnv* env, jobject, jlong handle, jshortArray pcm, jobjectArray hotword_tokens) {
  try {
    if (!pcm) throw std::invalid_argument("Missing PCM audio");
    const auto size = env->GetArrayLength(pcm);
    if (size < 1 || size > 480000) throw std::invalid_argument("Audio length must be 1..30 seconds at 16 kHz");
    std::vector<int16_t> samples(size);
    env->GetShortArrayRegion(pcm, 0, size, reinterpret_cast<jshort*>(samples.data()));
    if (env->ExceptionCheck()) return nullptr;
    std::vector<std::string> hints;
    if (hotword_tokens) {
      const auto count = env->GetArrayLength(hotword_tokens);
      if (count > 223) throw std::invalid_argument("Whisper hint exceeds context limit");
      for (jsize i = 0; i < count; ++i) {
        auto token = static_cast<jstring>(env->GetObjectArrayElement(hotword_tokens, i));
        if (!token) throw std::invalid_argument("Missing Whisper hint token");
        const char* value = env->GetStringUTFChars(token, nullptr);
        if (!value) return nullptr;
        hints.emplace_back(value);
        env->ReleaseStringUTFChars(token, value);
        env->DeleteLocalRef(token);
      }
    }
    return generate(env, get_model(handle), whisper_log_mel(samples.data(), samples.size()), hints);
  } catch (const std::exception& e) { fail(env, e); return nullptr; }
}

extern "C" JNIEXPORT jstring JNICALL
Java_kr_sonkkeut_android_KoreanWhisperNative_transcribeFeatures(JNIEnv* env, jobject, jlong handle, jfloatArray features) {
  try {
    if (!features || env->GetArrayLength(features) != 80 * 3000)
      throw std::invalid_argument("Expected float32 log-mel features [80,3000]");
    std::vector<float> mel(80 * 3000);
    env->GetFloatArrayRegion(features, 0, mel.size(), mel.data());
    if (env->ExceptionCheck()) return nullptr;
    return generate(env, get_model(handle), mel);
  } catch (const std::exception& e) { fail(env, e); return nullptr; }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_kr_sonkkeut_android_KoreanWhisperNative_logMel(JNIEnv* env, jobject, jshortArray pcm) {
  try {
    if (!pcm) throw std::invalid_argument("Missing PCM audio");
    const auto size = env->GetArrayLength(pcm);
    if (size < 1 || size > 480000) throw std::invalid_argument("Audio length must be 1..30 seconds at 16 kHz");
    std::vector<int16_t> samples(size);
    env->GetShortArrayRegion(pcm, 0, size, reinterpret_cast<jshort*>(samples.data()));
    if (env->ExceptionCheck()) return nullptr;
    const auto mel = whisper_log_mel(samples.data(), samples.size());
    auto output = env->NewFloatArray(mel.size());
    if (output) env->SetFloatArrayRegion(output, 0, mel.size(), mel.data());
    return output;
  } catch (const std::exception& e) { fail(env, e); return nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_kr_sonkkeut_android_KoreanWhisperNative_close(JNIEnv*, jobject, jlong handle) {
  std::shared_ptr<Whisper> model;
  { std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(handle);
    if (it == registry.end()) return;
    model = std::move(it->second); registry.erase(it);
  }
}
