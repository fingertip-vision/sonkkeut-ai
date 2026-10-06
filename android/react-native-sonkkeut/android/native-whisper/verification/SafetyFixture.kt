package kr.sonkkeut.asrsmoke

import android.content.Context
import android.media.AudioRecord
import android.util.AtomicFile
import kr.sonkkeut.android.KoreanWhisper
import kr.sonkkeut.android.WhisperModelInstaller
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.lang.reflect.InvocationTargetException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import java.util.zip.ZipException
import java.util.UUID

/** Real Android fixture. Only the dedicated APK's filesDir is written. No mocked model output. */
class SafetyFixture(private val context: Context) {
    private val checks = JSONArray()
    private val report = JSONObject().put("fixture", "synthetic-installer-and-lifecycle")
    private val fixtureRoot = File(context.filesDir, "safety-fixture").apply { mkdirs() }
    private val area = File(fixtureRoot, UUID.randomUUID().toString()).apply { mkdirs() }
    private val installer = WhisperModelInstaller(context)
    private val extract = installer.javaClass.getDeclaredMethod("extract", File::class.java,
        File::class.java, kotlin.jvm.functions.Function0::class.java).apply { isAccessible = true }
    private var whisper: KoreanWhisper? = null

    private fun checkpoint(name: String, detail: String = "") {
        checks.put(JSONObject().put("name", name).put("passed", true).put("detail", detail))
        report.put("checks", checks).put("status", "running")
        File(context.filesDir, "safety-result.json").writeText(report.toString(2), Charsets.UTF_8)
    }
    private fun unpack(zip: File, target: File, cancelled: () -> Boolean = { false }) {
        require(target.mkdir())
        try { extract.invoke(installer, zip, target, cancelled) }
        catch (error: InvocationTargetException) { throw error.targetException }
    }
    private fun zip(name: String, entries: List<Pair<String, ByteArray>>): File {
        val result = File(area, "$name.zip")
        ZipOutputStream(result.outputStream()).use { output ->
            for ((path, bytes) in entries) {
                output.putNextEntry(ZipEntry(path)); output.write(bytes); output.closeEntry()
            }
        }
        return result
    }
    private fun rejected(name: String, entries: List<Pair<String, ByteArray>>) {
        val archive = zip(name, entries)
        val target = File(area, "rejected-$name")
        var rejected = false
        try { unpack(archive, target) }
        catch (_: IllegalStateException) { rejected = true }
        catch (_: ZipException) { rejected = true } // Android 14+ may reject traversal before our own path guard.
        check(rejected) { "Unsafe ZIP accepted: $name" }
        checkpoint("zip_reject_$name")
    }
    private fun waitUntil(message: String, timeoutMs: Long = 120000, condition: () -> Boolean) {
        val deadline = System.nanoTime() + timeoutMs * 1000000
        while (!condition()) {
            check(System.nanoTime() < deadline) { "Timed out: $message" }
            Thread.sleep(25)
        }
    }
    private fun await(latch: CountDownLatch, name: String) {
        check(latch.await(120, TimeUnit.SECONDS)) { "Timed out: $name" }
    }
    fun run(): JSONObject {
        try {
            // This is an explicit fresh run in this isolated APK. Never delete another app's state.
            require(fixtureRoot.canonicalFile.parentFile == context.filesDir.canonicalFile)
            require(area.canonicalFile.parentFile == fixtureRoot.canonicalFile)
            require(area.listFiles()?.isEmpty() == true) { "Safety fixture was already run; use a new isolated fixture APK data directory." }
            check(!installer.isInstalled()) { "Safety fixture requires this dedicated APK to have no installed speech model." }
            val config = File("/data/local/tmp/sonkkeut-asr-smoke/model/config.json").readBytes()
            rejected("traversal", listOf("whisper-elder-v3/../escaped.json" to config))
            rejected("absolute", listOf("/escaped.json" to config))
            rejected("windows_absolute", listOf("C:\\escaped.json" to config))
            rejected("unknown_file", listOf("whisper-elder-v3/unknown.json" to config))
            rejected("nested", listOf("whisper-elder-v3/nested/config.json" to config))
            rejected("wrong_hash", listOf("whisper-elder-v3/config.json" to config.clone().also { it[0] = (it[0].toInt() xor 1).toByte() }))
            rejected("oversize", listOf("whisper-elder-v3/config.json" to (config + byteArrayOf(0))))
            rejected("missing", listOf("whisper-elder-v3/config.json" to config))
            rejected("duplicate_normalized", listOf("whisper-elder-v3/config.json" to config, "whisper-elder-v3\\config.json" to config))
            rejected("too_many_entries", (1..33).map { "directory$it/" to byteArrayOf() })
            check(!File(area, "escaped.json").exists())
            checkpoint("no_path_escape")
            var cancellationChecks = 0
            var cancelled = false
            try {
                unpack(File("/data/local/tmp/sonkkeut-asr-smoke/model.zip"), File(area, "cancelled-extraction")) { ++cancellationChecks > 8 }
            } catch (_: IllegalStateException) { cancelled = true }
            check(cancelled && cancellationChecks > 8)
            checkpoint("extract_cancel_during_stream", "Cancelled while reading the real archive, not a synthetic result.")

            // Invalid URL goes through the public install() entry point and its actual finally cleanup.
            var badUrl = false
            try { installer.install("http://127.0.0.1/model.zip", { false }) { _, _, _ -> } }
            catch (_: IllegalArgumentException) { badUrl = true }
            val installRoot = File(context.filesDir, "sonkkeut-asr")
            check(badUrl && !installer.isInstalled())
            check(installRoot.listFiles()?.none { it.name.endsWith(".part") } == true)
            check(File(installRoot, "versions").listFiles()?.isEmpty() == true)
            checkpoint("failed_install_cleanup_and_no_publication", "HTTP URL rejected before network; unpublished archive/staging removed.")
            val checkedUrl = installer.javaClass.getDeclaredMethod("checkedUrl", String::class.java).apply { isAccessible = true }
            for (value in listOf("https://user:password@example.invalid/model", "https://example.invalid:8443/model", "file:///data/model")) {
                var denied = false
                try { checkedUrl.invoke(installer, value) }
                catch (error: InvocationTargetException) { denied = error.targetException is IllegalArgumentException }
                check(denied)
            }
            checkpoint("url_reject_credentials_non443_nonhttps")

            // Verify every byte of the user's original archive payload with the production extraction code.
            val version = File(File(installRoot, "versions"), WhisperModelInstaller.ARCHIVE_SHA256)
            unpack(File("/data/local/tmp/sonkkeut-asr-smoke/model.zip"), version)
            check(version.listFiles()?.size == 5)
            checkpoint("original_five_files_size_and_sha256", "Actual model.bin and four metadata files, strict per-file digest validation.")

            // Seed only the atomic pointer after real verification; no download/atomic-install claim is made.
            val current = AtomicFile(File(installRoot, "current.json"))
            val marker = JSONObject().put("archive_sha256", WhisperModelInstaller.ARCHIVE_SHA256).put("version", "v3").toString().toByteArray()
            val stream = current.startWrite(); stream.write(marker); current.finishWrite(stream)
            check(installer.modelDirectory()?.canonicalFile == version.canonicalFile)
            val interrupted = current.startWrite(); interrupted.write("{incomplete".toByteArray()); interrupted.fd.sync()
            check(installer.modelDirectory()?.canonicalFile == version.canonicalFile)
            current.failWrite(interrupted)
            checkpoint("atomic_pointer_interrupted_write_reads_previous", "Production reader plus Android AtomicFile; download publication was fixture-seeded.")

            val engine = KoreanWhisper(context).also { whisper = it }
            engine.setMenuContext(listOf("아메리카노", "카페라떼"))
            val ready = CountDownLatch(1)
            val preparationError = AtomicReference<Throwable?>()
            engine.prepare({ ready.countDown() }, { preparationError.set(it); ready.countDown() })
            await(ready, "model prepare"); preparationError.get()?.let { throw it }
            waitUntil("prepare idle") { engine.status()["busy"] == false }
            check(engine.status()["ready"] == true && engine.status()["installed"] == true)
            checkpoint("wrapper_prepare_actual_installed_model")

            // Real AudioRecord must enter RECORDSTATE_RECORDING before cancellation is asserted.
            val staleCallbacks = AtomicInteger()
            val recordingField = engine.javaClass.getDeclaredField("recording").apply { isAccessible = true }
            @Suppress("UNCHECKED_CAST")
            val recording = recordingField.get(engine) as AtomicReference<AudioRecord?>
            engine.listen({ staleCallbacks.incrementAndGet() }, { staleCallbacks.incrementAndGet() })
            waitUntil("AudioRecord recording", 10000) { recording.get()?.recordingState == AudioRecord.RECORDSTATE_RECORDING }
            Thread.sleep(250)
            engine.cancel()
            waitUntil("cancel recording idle", 10000) { engine.status()["busy"] == false }
            Thread.sleep(150)
            check(recording.get() == null && staleCallbacks.get() == 0)
            checkpoint("actual_audio_record_cancel_suppresses_callbacks", "Silent API35 emulator capture, not a physical microphone quality test.")

            val bytes = File("/data/local/tmp/sonkkeut-asr-smoke/order.raw").readBytes()
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
            val pcm = ShortArray(buffer.remaining()).also { buffer.get(it) }
            val cancelledInferenceCallbacks = AtomicInteger()
            engine.transcribePcm(pcm, { cancelledInferenceCallbacks.incrementAndGet() }, { cancelledInferenceCallbacks.incrementAndGet() })
            Thread.sleep(750)
            check(engine.status()["busy"] == true)
            engine.cancel()
            val overlapping = CountDownLatch(1)
            val overlapErrors = AtomicInteger()
            engine.transcribePcm(pcm, { overlapping.countDown() }, { if (it is IllegalStateException) overlapErrors.incrementAndGet(); overlapping.countDown() })
            await(overlapping, "busy request rejection")
            check(overlapErrors.get() == 1)
            waitUntil("cancel inference idle") { engine.status()["busy"] == false }
            Thread.sleep(100)
            check(cancelledInferenceCallbacks.get() == 0)
            checkpoint("cancel_active_inference_and_reject_busy_new_request", "CT2 finishes worker inference; invalidated callbacks stay suppressed.")

            val completed = CountDownLatch(1)
            val result = AtomicReference<KoreanWhisper.Result?>()
            val inferenceError = AtomicReference<Throwable?>()
            engine.transcribePcm(pcm, { result.set(it); completed.countDown() }, { inferenceError.set(it); completed.countDown() })
            await(completed, "new request after cancellation"); inferenceError.get()?.let { throw it }
            val actual = result.get() ?: error("No result after cancellation")
            check(actual.text == "따뜻한 아메리카노 두 잔 하고 카페라떼 한 잔 포장해 주세요.")
            check(actual.inferenceMs > 0 && actual.provider == "ct2-whisper-v3")
            report.put("new_request_text", actual.text).put("inference_ms", actual.inferenceMs)
            checkpoint("new_request_after_cancel_uses_real_contextual_model")
            waitUntil("completed idle") { engine.status()["busy"] == false }
            engine.close()
            val closed = CountDownLatch(1)
            val closeErrors = AtomicInteger()
            engine.transcribePcm(pcm, { closed.countDown() }, { if (it is IllegalStateException) closeErrors.incrementAndGet(); closed.countDown() })
            await(closed, "closed request rejection")
            check(closeErrors.get() == 1)
            checkpoint("closed_engine_rejects_new_request")
            report.put("status", "completed").put("passed", checks.length())
        } catch (error: Throwable) {
            report.put("status", "failed").put("error_class", error.javaClass.name).put("error", error.message)
        } finally { whisper?.close() }
        return report.put("checks", checks)
    }
}
