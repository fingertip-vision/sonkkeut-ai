# Android model installer and lifecycle fixture

Dedicated package: `kr.sonkkeut.asrsmoke`. API35 x86_64 emulator executing ARM64 libraries via native translation. This is actual model inference, not a physical ARM64 device performance measurement.

Run the prototype APK with `--es mode safety` after microphone permission has been granted. Its private `safety-fixture` directory receives synthetic malformed ZIPs. No `com.sonkkeut` state or personal apps are accessed. Run once with no installed model in this dedicated APK's data directory. Source: `smoke/SafetyFixture.kt`; report collector: `collect_safety.py`.

Required read-only public model/reference fixtures on the emulator:

- `/data/local/tmp/sonkkeut-asr-smoke/model.zip`: original ASR v3 archive, SHA256 `d6d5b3c3efbc7be6b9d6585fca8155e4418fa00f443eeb9aff464075da37c3ea`.
- `/data/local/tmp/sonkkeut-asr-smoke/model/config.json`: original configuration used to build malformed ZIPs.
- `/data/local/tmp/sonkkeut-asr-smoke/order.raw`: synthetic Korean order, PCM16LE mono 16 kHz, 88141 samples.

Observed result: **21/21 passed**, saved as `fixture/android-safety-result.json`.

Checks cover traversal, absolute/Windows paths, unknown/nested files, wrong file digest, oversized file, missing files, normalized duplicate name, more than 32 entries, no path escape, cancellation while extracting the original archive, rejected HTTP installation cleanup, URL credential/non443/nonHTTPS rejection, real five-file exact size/SHA256 verification, interrupted atomic-pointer write retaining the previous version, actual model preparation, real AudioRecord entering recording state then cancellation with zero stale callbacks, cancellation of active CT2 inference with busy overlapping-request rejection, successful new actual inference after cancellation, and rejection after engine close.

The new request decoded `따뜻한 아메리카노 두 잔 하고 카페라떼 한 잔 포장해 주세요.` with menu context. CT2 reported 20101 ms under ARM64 translation. AudioRecord used the emulator's silent input; this does not assess a physical microphone. CT2 cancellation invalidates callbacks and waits for the active native call to finish; it does not interrupt individual decoder steps.

Scope: successful extraction invokes the production private extractor via reflection and verifies the actual 483 MB model payload. The install failure test uses the public installer and its cleanup. The valid pointer is fixture-seeded only after actual extraction verification. The interrupted-pointer test invokes Android AtomicFile and the production model-directory reader. The fixture does **not** claim a full successful network download or atomic installation publication test.

The four production Kotlin files, two native libraries, and frozen source bundle are unchanged by these tests. The React Native owner independently corrected its four operation-entry guards to include all pending speech, download, and prepare promises; this prevents generation changes from suppressing an earlier queued terminal callback.
