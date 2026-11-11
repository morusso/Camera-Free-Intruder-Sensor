# Data collection protocol

This is the checklist for recording training and test data. Follow it every time, so that sessions are comparable and the labels can be trusted. A model is only as good as its data, and most of the project's time goes here.

## The four classes

| Class | What is happening in the room |
|---|---|
| `empty` | Nobody is in the room. Nothing is moving. |
| `static_person` | One person stands or sits still (small natural movements are fine). |
| `walking` | One person moves around: walking, entering, leaving. |
| `interference` | Nobody is in the room, but something else changes the radio channel: a fan, air conditioning, a door opening, a device switching on, someone moving behind the wall. |

`interference` is a real class, not noise to be ignored. If the model never sees a fan during training, it will raise an alarm for every fan in production.

## What to record, per scenario

| Scenario | How | Length per take | Repeats |
|---|---|---|---|
| `empty` | Leave the room, close the door, start the recording from outside. | 10 min | At different times of day (morning, noon, evening, night) |
| `static_person` | Stand or sit still at one marked position. Positions: on the TX-RX line, 1 m off the line (both sides), near TX, near RX, in a far corner. At least 5 positions, both standing and sitting. | 2 min | Each position at least 3 times, with at least 2 different people |
| `walking` | Walk a fixed route: across the link, along the link, around the room. Slow, normal and fast pace. Include entering and leaving through the door. | 2 min | Each route and pace at least 3 times, with at least 2 different people |
| `interference` | One source at a time: fan (each speed), air conditioning on/off, door opening and closing without entering, a device switching on (monitor, kettle, charger), someone walking behind the wall. | 3 min | Each source at least 3 times |

Target amount before Stage 2 is done: **at least 3 hours per class in total**, collected over **at least 5 different days**. Less than that and the test numbers will not mean much.

## Rules that keep the data honest

1. **One scenario per take.** Do not switch from walking to standing in the middle of a take without pressing the matching label button (see Labels).
2. **Only one person in the room** unless the scenario explicitly says otherwise. Anyone else stays out of the room, and out of the line between the boards.
3. **Do not touch the boards or antennas** during a take. If something is moved, start a new session.
4. **Keep the setup fixed within a session.** Boards, antennas, furniture and doors stay where they are.
5. **Write down anything unusual** (someone walked past, a phone rang, the router restarted) in the session notes. Doubtful stretches can then be marked uncertain or dropped.
6. **Start and stop with margin.** The first and last few seconds of a take are unreliable (you are walking to or from your place). The alignment script cuts 2 seconds around every label change.
7. **Sync the computer clock with NTP before every session** (macOS: System Settings > General > Date & Time > set time automatically). The labels are matched to the CSI by time, so a clock that is off by one second shifts every label.

## Labels

Labels are recorded on a phone with the labeling page (step 2.2). The buttons are:

| Button | Becomes class | Meaning |
|---|---|---|
| empty | `empty` | The room is empty from now on. |
| standing | `static_person` | The person is standing or sitting still from now on. |
| walking | `walking` | The person is walking from now on. |
| entering | `walking` | The person is entering, until they reach their place. |
| leaving | `walking` | The person starts leaving, until the room is empty. |
| interference | `interference` | An interference source is active (the source is written in the note). |

Each press writes a line to the log: time, label, `start` or `end`. The 2 seconds around every change are marked uncertain.

## Sessions and files

Every session gets its own folder:

```
data/raw/YYYY-MM-DD_<short-description>/
    csi_*.npz        recordings from csi_reader.py
    labels.csv       log from the labeling page
    meta.json        description of the setup (see below)
```

`meta.json` example:

```json
{
  "date": "2026-11-12",
  "room": "office-2",
  "channel": 11,
  "send_rate_hz": 100,
  "htltf": false,
  "tx_position": "window side, 1.2 m high, on a shelf",
  "rx_position": "door side, 1.2 m high, on a shelf",
  "link_distance_m": 4.5,
  "people": ["A"],
  "furniture_changes": "none",
  "notes": "AC off all session"
}
```

The folder `data/` is not in git. Back it up separately.

## Splitting the data

Never split randomly by window. Neighbouring windows are almost identical, so a random split leaks the answer from train to test and the score looks much better than it is.

- **Train / validation:** split by session.
- **Test, part 1:** whole days that were never used for training or tuning.
- **Test, part 2:** a session recorded after moving a piece of furniture (for example the shelf or a chair), to measure how the model copes with a changed room.
- Decide the split before looking at results and write it down in `docs/`.

## Checklist for one session

- [ ] Clock synced with NTP
- [ ] Boards and antennas in place, positions measured and written to `meta.json`
- [ ] Recording started (`csi_reader.py`), heatmap shows a stable signal
- [ ] Labeling page open on the phone
- [ ] Takes recorded according to the table above
- [ ] Recording stopped, `jitter_report.py` run: no large loss or jitter problems
- [ ] `meta.json` and notes saved in the session folder
