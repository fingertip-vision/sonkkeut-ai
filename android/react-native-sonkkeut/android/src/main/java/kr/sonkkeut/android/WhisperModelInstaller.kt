package kr.sonkkeut.android

import android.content.Context
import android.util.AtomicFile
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.UUID
import java.util.zip.ZipInputStream

/** Blocking installer; call on the ASR worker. No weights are bundled in the APK. */
class WhisperModelInstaller(context: Context) {
    private val root = File(context.filesDir, "sonkkeut-asr")
    private val versions = File(root, "versions")
    private val current = AtomicFile(File(root, "current.json"))

    fun isInstalled() = modelDirectory() != null
    fun modelDirectory(): File? = runCatching {
        val record = JSONObject(current.readFully().toString(Charsets.UTF_8))
        if (record.getString("archive_sha256") != ARCHIVE_SHA256) return null
        val directory = File(versions, ARCHIVE_SHA256)
        if (!FILES.all { (name, spec) -> File(directory, name).let { it.isFile && it.length() == spec.size } }) return null
        directory
    }.getOrNull()

    fun install(downloadUrl: String = DOWNLOAD_URL, cancelled: () -> Boolean, progress: (Long, Long, String) -> Unit) {
        if (isInstalled()) return
        check(root.mkdirs() || root.isDirectory)
        check(versions.mkdirs() || versions.isDirectory)
        val attempt = UUID.randomUUID().toString()
        val archive = File(root, "$attempt.zip.part")
        val stage = File(versions, ".$attempt.staging")
        check(stage.mkdir())
        try {
            download(downloadUrl, archive, cancelled, progress)
            check(!cancelled()) { "모델 다운로드를 중지했습니다." }
            progress(0, FILES.values.sumOf { it.size }, "installing")
            extract(archive, stage, cancelled)
            check(!cancelled()) { "모델 설치를 중지했습니다." }
            val target = File(versions, ARCHIVE_SHA256)
            if (target.exists()) {
                // Only an unpublished incomplete version of this exact digest can be replaced.
                check(target.canonicalFile.parentFile == versions.canonicalFile)
                check(target.deleteRecursively())
            }
            check(stage.renameTo(target)) { "모델 폴더를 설치하지 못했습니다." }
            // Immutable, verified directory is fully prepared before this atomic pointer changes.
            val data = JSONObject().put("archive_sha256", ARCHIVE_SHA256).put("version", "v3")
            val stream = current.startWrite()
            try { stream.write(data.toString().toByteArray(Charsets.UTF_8)); current.finishWrite(stream) }
            catch (error: Throwable) { current.failWrite(stream); throw error }
            progress(FILES.values.sumOf { it.size }, FILES.values.sumOf { it.size }, "ready")
        } finally { archive.delete(); if (stage.exists()) stage.deleteRecursively() }
    }

    private fun download(url: String, destination: File, cancelled: () -> Boolean, progress: (Long, Long, String) -> Unit) {
        var next = checkedUrl(url)
        var connection: HttpURLConnection? = null
        repeat(6) {
            if (connection == null) {
                val candidate = next.openConnection() as HttpURLConnection
                candidate.instanceFollowRedirects = false
                candidate.connectTimeout = 15000
                candidate.readTimeout = 30000
                candidate.setRequestProperty("Accept", "application/octet-stream")
                val status = candidate.responseCode
                if (status in listOf(301, 302, 303, 307, 308)) {
                    val location = candidate.getHeaderField("Location") ?: error("모델 다운로드 주소가 없습니다.")
                    next = checkedUrl(URL(next, location).toExternalForm())
                    candidate.disconnect()
                } else {
                    check(status == 200) { candidate.disconnect(); "모델 다운로드에 실패했습니다 ($status)." }
                    connection = candidate
                }
            }
        }
        val active = connection ?: error("모델 다운로드 주소 이동이 너무 많습니다.")
        try {
            val total = active.contentLengthLong
            check(total < 600L * 1024 * 1024) { "모델 파일이 너무 큽니다." }
            val digest = MessageDigest.getInstance("SHA-256")
            var bytes = 0L
            var lastProgress = 0L
            active.inputStream.use { input -> FileOutputStream(destination).use { output ->
                val buffer = ByteArray(64 * 1024)
                while (true) {
                    check(!cancelled()) { "모델 다운로드를 중지했습니다." }
                    val count = input.read(buffer)
                    if (count < 0) break
                    bytes += count
                    check(bytes <= 600L * 1024 * 1024) { "모델 파일이 너무 큽니다." }
                    output.write(buffer, 0, count); digest.update(buffer, 0, count)
                    val now = System.currentTimeMillis()
                    if (now - lastProgress >= 250) { progress(bytes, total, "downloading"); lastProgress = now }
                }
                output.fd.sync()
            } }
            check(total < 0 || bytes == total) { "모델 파일 다운로드가 끊겼습니다." }
            check(digest.digest().hex() == ARCHIVE_SHA256) { "모델 파일 검증에 실패했습니다." }
            progress(bytes, bytes, "verifying_archive")
        } finally { active.disconnect() }
    }

    private fun extract(archive: File, destination: File, cancelled: () -> Boolean) {
        val seen = mutableSetOf<String>()
        ZipInputStream(FileInputStream(archive)).use { zip ->
            var entries = 0
            while (true) {
                val entry = zip.nextEntry ?: break
                check(++entries <= 32) { "모델 압축 파일 항목이 너무 많습니다." }
                val path = entry.name.replace('\\', '/')
                check(!path.startsWith('/') && ':' !in path && '\u0000' !in path)
                check(path.split('/').none { it == ".." || it == "." }) { "안전하지 않은 모델 압축 경로입니다." }
                if (entry.isDirectory) { zip.closeEntry(); continue }
                check(path.startsWith("whisper-elder-v3/") && path.count { it == '/' } == 1) { "알 수 없는 모델 압축 경로입니다." }
                val name = path.substringAfter('/')
                val spec = FILES[name] ?: error("알 수 없는 모델 파일입니다.")
                check(seen.add(name)) { "중복 모델 파일입니다." }
                val outputFile = File(destination, name)
                check(outputFile.canonicalFile.parentFile == destination.canonicalFile)
                var bytes = 0L
                val digest = MessageDigest.getInstance("SHA-256")
                FileOutputStream(outputFile).use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        check(!cancelled()) { "모델 설치를 중지했습니다." }
                        val count = zip.read(buffer)
                        if (count < 0) break
                        bytes += count
                        check(bytes <= spec.size) { "모델 파일 크기가 올바르지 않습니다." }
                        output.write(buffer, 0, count); digest.update(buffer, 0, count)
                    }
                    output.fd.sync()
                }
                check(bytes == spec.size && digest.digest().hex() == spec.sha256) { "모델 파일 검증에 실패했습니다." }
                zip.closeEntry()
            }
        }
        check(seen == FILES.keys) { "모델 파일이 누락됐습니다." }
    }

    private fun checkedUrl(value: String): URL {
        val url = URL(value)
        require(url.protocol == "https" && url.userInfo == null && url.host.isNotBlank() && url.port in listOf(-1, 443)) {
            "모델 다운로드에는 인증 정보 없는 HTTPS 주소가 필요합니다."
        }
        return url
    }
    private fun ByteArray.hex() = joinToString("") { "%02x".format(it.toInt() and 255) }
    private data class Spec(val size: Long, val sha256: String)
    companion object {
        const val DOWNLOAD_URL = "https://github.com/fingertip-vision/sonkkeut-ai/releases/download/asr-whisper-elder-v3/whisper-elder-v3-ct2.zip"
        const val ARCHIVE_SHA256 = "d6d5b3c3efbc7be6b9d6585fca8155e4418fa00f443eeb9aff464075da37c3ea"
        private val FILES = linkedMapOf(
            "config.json" to Spec(2609, "364e66f09ce360358f61e549786419eedc8e6ab709186304dc77f84a3c4966c8"),
            "model.bin" to Spec(483546977, "b5468a47157fad449fe6ff093786848badbd7c0c1b28aa15c3dd346ab8a365da"),
            "preprocessor_config.json" to Spec(329, "3e4dc53c73e7e3954fc4435034f5027733c61d7c123b38bbd1461598fd8503ef"),
            "tokenizer_config.json" to Spec(2253, "852662bea40ac4b9e200eb6c56c688f4f7420c68760e5c4fe7cc4d082234e451"),
            "vocabulary.json" to Spec(1119969, "aa4e6188766a36f6a7f3e44db3f74823c6fee0bf33a9559247e18fedb3bc8d55")
        )
    }
}
