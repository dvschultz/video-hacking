# Multi-Stem Analysis Guide

## Quick Start - Just "Other" Track (Your Use Case)

```bash
# Option 1: Let script handle separation
python src/multi_stem_analysis.py \
  --audio data/input/song.mp3 \
  --output data/cuts.json \
  --stems other \
  --methods spectral \
  --min-gaps 0.15

# Option 2: Use pre-separated stems (faster for iteration)
demucs -n htdemucs data/input/song.mp3 -o data/separated

python src/multi_stem_analysis.py \
  --audio data/input/song.mp3 \
  --output data/cuts.json \
  --separated-dir data/separated/htdemucs/song \
  --stems other \
  --methods spectral \
  --min-gaps 0.15 \
  --skip-separation
```

## Advanced: Combining Multiple Stems

### Example 1: Other + Bass for Varied Rhythm

```bash
python src/multi_stem_analysis.py \
  --audio data/input/song.mp3 \
  --output data/cuts_other_bass.json \
  --stems other bass \
  --methods spectral default \
  --min-gaps 0.15 0.25 \
  --merge-strategy union \
  --proximity 0.05
```

**What this does:**
- `other` with spectral method → catches melodic/harmonic changes
- `bass` with default method → catches groove changes
- `union` strategy → combines all onsets
- `proximity 0.05` → merges onsets within 50ms (avoids double-cuts)

### Example 2: Sparse Drums + Dense Other

```bash
python src/multi_stem_analysis.py \
  --audio data/input/song.mp3 \
  --output data/cuts_layered.json \
  --stems drums other \
  --methods energy spectral \
  --min-gaps 0.5 0.1 \
  --merge-strategy union
```

**What this does:**
- `drums` with min-gap 0.5s → only major drum hits
- `other` with min-gap 0.1s → detailed melodic cuts
- Creates layered rhythm: major accents + intricate details

### Example 3: Only Major Changes (Conservative)

```bash
python src/multi_stem_analysis.py \
  --audio data/input/song.mp3 \
  --output data/cuts_conservative.json \
  --stems drums bass other \
  --methods energy default spectral \
  --min-gaps 0.2 0.2 0.2 \
  --merge-strategy intersection
```

**What this does:**
- Analyzes all three stems
- `intersection` → only keeps onsets that appear in multiple stems
- Result: Only major musical events where multiple elements change together

## Merging Strategies Explained

### Union (Most Cuts)
```bash
--merge-strategy union --proximity 0.05
```
- Combines ALL onsets from all stems
- Removes duplicates within 50ms
- **Use when:** You want rich, varied cuts from multiple sources
- **Result:** Most onset times, layered rhythm

### Intersection (Fewest Cuts)
```bash
--merge-strategy intersection --proximity 0.1
```
- Only keeps onsets that appear in multiple stems (within 100ms of each other)
- **Use when:** You want only major musical moments
- **Result:** Conservative, only significant changes

### Primary Plus (Hybrid)
```bash
--merge-strategy primary_plus --proximity 0.1
```
- Uses first stem as the base
- Adds onsets from other stems only if they don't create dense clusters
- **Use when:** One stem is primary, others are accents
- **Result:** Structured rhythm with occasional variation

### Weighted (Advanced)
```bash
--merge-strategy weighted
```
- Currently treats all equally, but you can modify the code to weight stems
- **Use when:** You want algorithmic blending with priorities

## Per-Stem Best Practices

### Drums
- **Method:** `energy` (detects percussive hits)
- **Min-gap:** 0.1-0.5s depending on density desired
- **Use for:** Regular rhythm, beat-based cuts

### Bass
- **Method:** `default` or `energy`
- **Min-gap:** 0.2-0.4s
- **Use for:** Groove changes, low-frequency accents

### Other
- **Method:** `spectral` or `combined`
- **Min-gap:** 0.1-0.2s
- **Use for:** Melodic/harmonic changes, irregular rhythm (YOUR CHOICE)

### Vocals
- **Method:** `default` or `spectral`
- **Min-gap:** 0.3-0.6s
- **Use for:** Lyrical phrases, vocal emphasis

## Output Format

The script outputs JSON with both merged and individual stem data:

```json
{
  "audio_file": "data/input/song.mp3",
  "num_onsets": 187,
  "onset_times": [0.12, 0.58, 1.02, ...],
  "segment_durations": [0.46, 0.44, ...],
  "stems": {
    "other": {
      "num_onsets": 145,
      "onset_times": [...],
      "method": "spectral",
      "stats": {...}
    },
    "bass": {
      "num_onsets": 78,
      "onset_times": [...],
      "method": "default",
      "stats": {...}
    }
  }
}
```

## Iteration Workflow

1. **Separate once:**
   ```bash
   demucs -n htdemucs data/input/song.mp3 -o data/separated
   ```

2. **Try different combinations:**
   ```bash
   # Just other (your preference)
   python src/multi_stem_analysis.py --audio data/input/song.mp3 \
     --separated-dir data/separated/htdemucs/song --skip-separation \
     --stems other --output data/cuts_other.json

   # Add bass for comparison
   python src/multi_stem_analysis.py --audio data/input/song.mp3 \
     --separated-dir data/separated/htdemucs/song --skip-separation \
     --stems other bass --output data/cuts_other_bass.json

   # Conservative (intersection)
   python src/multi_stem_analysis.py --audio data/input/song.mp3 \
     --separated-dir data/separated/htdemucs/song --skip-separation \
     --stems other bass drums --merge-strategy intersection \
     --output data/cuts_conservative.json
   ```

3. **Compare results** and pick the one that fits your artistic vision!

## Why Combine Stems?

- **Single stem (other):** Clean, irregular rhythm following melody/harmony
- **Two stems (other + bass):** Adds low-frequency groove changes
- **Three stems (other + bass + drums):** Full musical spectrum, busier
- **Intersection of multiple:** Only major structural changes

**For your stated goal** (irregular, non-regular sequences), **stick with just "other"** using spectral method. Combining would add more regularity from drums/bass.
