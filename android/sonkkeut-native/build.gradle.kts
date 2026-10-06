import java.io.File
import java.net.URL
import java.security.MessageDigest

plugins { id("com.android.library"); id("org.jetbrains.kotlin.android") }
android {
    namespace = "kr.sonkkeut.nativeai"
    compileSdk = 35
    defaultConfig { minSdk = 26; consumerProguardFiles("../react-native-sonkkeut/android/consumer-rules.pro") }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
    sourceSets.getByName("main") {
        java.setSrcDirs(listOf("src/main/java", "../react-native-sonkkeut/android/src/main/java/kr/sonkkeut/core", "../react-native-sonkkeut/android/src/main/java/kr/sonkkeut/android"))
        assets.srcDir("../react-native-sonkkeut/android/src/main/assets")
        jniLibs.srcDir("../react-native-sonkkeut/android/src/main/jniLibs")
    }
    androidResources { noCompress += listOf("onnx", "task") }
}
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
    implementation("org.opencv:opencv:4.10.0")
    implementation("com.google.mlkit:text-recognition-korean:16.0.1")
}
val handModel = file("../react-native-sonkkeut/android/src/main/assets/sonkkeut/hand_landmarker.task")
val downloadHandModel = tasks.register("downloadHandModel") {
    outputs.file(handModel)
    doLast {
        val expected = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
        if (!handModel.exists()) {
            handModel.parentFile.mkdirs()
            val temporary = File(handModel.parentFile,"hand_landmarker.download")
            try {
                URL("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task").openStream().use { input -> temporary.outputStream().use { input.copyTo(it) } }
                val digest = MessageDigest.getInstance("SHA-256").digest(temporary.readBytes()).joinToString("") { "%02x".format(it) }
                check(digest == expected) { "손끝 모델 파일의 검증값이 일치하지 않습니다." }
                temporary.copyTo(handModel,overwrite=false)
            } finally { temporary.delete() }
        }
        check(MessageDigest.getInstance("SHA-256").digest(handModel.readBytes()).joinToString("") { "%02x".format(it) } == expected)
    }
}
tasks.named("preBuild") { dependsOn(downloadHandModel) }
