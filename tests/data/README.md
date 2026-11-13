# Test data

`real_pattern.npz` and `real_startup.npz` hold device callback times and frame counts measured on the development PC
(`tools/audio_soak.py`): numbers only, no audio, no device names. `headset_pattern.npz` holds the same for the first
120 s of two real soaks on a 16 kHz Bluetooth hands-free headset mic, which loses a few ms of audio at a time
(engine-obs review r1, B1): per run, the mic and cable callbacks (entry time, PortAudio time, frames), the worker
steps' times and the two rates, with times relative to when the audio came on.

The two recordings below are real audio for the J06 speech tests (`tests/speech.py`). Each was cut and converted with
`tools/convert_fixture.py` (the `tools` dependency group): channels averaged, resampled exactly from 44.1 to 48 kHz
(`scipy.signal.resample_poly`, 160/147), rounded to 16 bits. No gain, filtering or fades were applied.

## speech.wav

One woman reading aloud, with natural pauses between phrases, soft word endings and fricatives.

- Title: *The Invisible Man* (solo), chapter 5, by H. G. Wells, read for LibriVox
- Author: Cathy Barratt (reader), LibriVox
- Source: https://archive.org/details/invisibleman_1209_librivox_cb
- File: `invisibleman_05_wells.mp3` (the uploaded original, 44.1 kHz mono VBR MP3)
- Licence: Public Domain (LibriVox recordings are public domain; the item carries the Public Domain Mark 1.0,
  http://creativecommons.org/publicdomain/mark/1.0/)
- Conversion: 30.5 s from 42.5 s into the file; 48 kHz, mono, 16-bit
  (`python tools/convert_fixture.py invisibleman_05_wells.mp3 tests/data/speech.wav --start 42.5 --seconds 30.5`)

## noise.wav

Real room noise: the steady hum of an air conditioner.

- Title: "air conditioner hum"
- Author: Gravity Sound (https://www.gravitysound.studio/)
- Source: https://commons.wikimedia.org/wiki/File:Air_conditioner_hum_(Gravity_Sound).wav
- Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Attribution: "air conditioner hum" by Gravity
  Sound, converted to 48 kHz mono 16-bit for Clean Voice's tests.
- Conversion: 13.0 s from 1.0 s into the file, leaving out the recording's 0.5 s fade-in (looped, it put a 15 dB
  dip into the "room" every 14 s); stereo averaged to mono; 48 kHz, 16-bit
  (`python tools/convert_fixture.py ac_hum.wav tests/data/noise.wav --start 1.0 --seconds 13`)

## fan.wav

Real broadband room noise: an electric fan. The hum above is tonal (54.7 Hz and its harmonics), and RNNoise reports it
as voice for up to 0.76 s at a time; this second noise checks the chain on the steady broadband hiss of a fan or PC.

- Title: "fan"
- Author: Gravity Sound (https://www.gravitysound.studio/)
- Source: https://commons.wikimedia.org/wiki/File:Fan_(Gravity_Sound).wav
- Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Attribution: "fan" by Gravity Sound, converted
  to 48 kHz mono 16-bit for Clean Voice's tests.
- Conversion: 13.0 s from 2.0 s into the file (the end fades out); stereo averaged to mono; 48 kHz, 16-bit
  (`python tools/convert_fixture.py fan.wav tests/data/fan.wav --start 2.0 --seconds 13`)
