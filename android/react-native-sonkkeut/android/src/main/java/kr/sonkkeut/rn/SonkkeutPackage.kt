package kr.sonkkeut.rn

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager
import com.mrousavy.camera.frameprocessors.FrameProcessorPluginRegistry

/** React Native 자동 연결(autolinking)이 찾는 패키지 */
class SonkkeutPackage : ReactPackage {
    companion object {
        init {
            FrameProcessorPluginRegistry.addFrameProcessorPlugin("sonkkeut") { proxy, options ->
                SonkkeutFrameProcessor(proxy, options)
            }
        }
    }

    override fun createNativeModules(ctx: ReactApplicationContext): List<NativeModule> = listOf(SonkkeutModule(ctx))

    override fun createViewManagers(ctx: ReactApplicationContext): List<ViewManager<*, *>> = emptyList()
}
