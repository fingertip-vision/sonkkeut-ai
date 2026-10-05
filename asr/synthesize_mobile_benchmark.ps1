$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$speechBenchRoot = Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$speechBenchData = Get-Content -Raw -Encoding UTF8 (Join-Path $speechBenchRoot 'outputs/sonkkeut-ai/asr/data/kiosk_speech_cases.json') | ConvertFrom-Json
$speechBenchWav = Join-Path $speechBenchRoot 'work/speech-benchmark/audio'
New-Item -ItemType Directory -Path $speechBenchWav -Force | Out-Null
$speechBenchSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speechBenchSynth.SelectVoice('Microsoft Heami Desktop')
$speechBenchFormat = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
try {
    foreach ($case in $speechBenchData.cases) {
        if ($case.signal) { continue }
        $speechBenchSynth.Rate = $case.rate
        $speechBenchSynth.SetOutputToWaveFile((Join-Path $speechBenchWav ('{0:D3}.wav' -f [int]$case.id)), $speechBenchFormat)
        $speechBenchSynth.Speak($case.text)
    }
    $speechBenchSynth.SetOutputToNull()
} finally { $speechBenchSynth.Dispose() }
Write-Output 'Generated Korean benchmark audio.'
