# Checkpoints

Copy trained model checkpoints here.

**4-stem model** (vocals, drums, bass, other):

```bash
mkdir -p spectrogram_stems
cp /path/to/cap/outputs/spectrogram_stems/best.pt spectrogram_stems/
```

**Vocals-only model** (vocals + accompaniment):

```bash
mkdir -p spectrogram
cp /path/to/cap/outputs/spectrogram/best.pt spectrogram/
```

The API expects:
- `checkpoints/spectrogram_stems/best.pt` for `?model=stems`
- `checkpoints/spectrogram/best.pt` for `?model=vocals`
