Add-Type -AssemblyName System.Speech
$utts = Get-Content -Raw -Encoding UTF8 "D:\sonkkeutgil\core\data\utt100.json" | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice("Microsoft Heami Desktop")
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$rates = @(-3, -2, -1, 0, 0, 1, 2)
foreach ($u in $utts) {
  $s.Rate = $rates[$u.id % $rates.Count]
  $f = "D:\sonkkeutgil\core\data\utt100_wav\clean\{0:D3}.wav" -f [int]$u.id
  $s.SetOutputToWaveFile($f, $fmt); $s.Speak($u.text)
}
$s.SetOutputToNull()
"done"
