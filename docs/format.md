# Receiver data format (RX → computer)

This is the agreement between the receiver firmware and the program on the computer: what the receiver sends over the USB cable when it gets CSI data. If we change the format on one side, we must change it on the other too.

Format version: **1** (draft). We will check it on real hardware in Stage 1 and fix it if anything does not match.

## How it works in short

The receiver gets a radio frame from the transmitter and sends **one line of text** to the computer. One line is one received frame. A line looks like this:

```
CSI_DATA,1042,1040,aa:bb:cc:dd:ee:ff,-48,11,-92,6,5123456,64,1,128,0,"[3,-5,2,7,...]"
```

The line starts with the word `CSI_DATA`, so the program can tell data apart from ordinary diagnostic messages. Then come fields separated by commas. At the end, in quotes and square brackets, is a long list of numbers: the CSI measurement itself.

## What each field means

The fields come in this order:

| Field | What it is |
|---|---|
| `rx_idx` | Sequence number of the frame, counted by the receiver. A gap in this number means data was lost on the way through the USB cable. |
| `tx_seq` | Sequence number of the frame, counted by the transmitter and written into the frame. A gap in this number means a frame was lost in the air. Two separate numbers let us tell exactly where data is lost. |
| `mac` | Hardware address of the transmitter. |
| `rssi` | Received signal strength in dBm (e.g. -48). The closer to zero, the stronger. |
| `rate` | Transmission rate the frame was sent at (ESP-IDF code). |
| `noise_floor` | Background noise level in dBm. |
| `channel` | Wi-Fi channel (e.g. 1, 6, 11). |
| `local_timestamp` | Time the frame was received, in microseconds since the chip clock started. It wraps around to zero roughly every 72 minutes, so the program on the computer must handle that. |
| `sig_len` | Frame length according to the radio header. |
| `rx_format` | Type of radio frame (ESP-IDF code). |
| `len` | How many numbers are in the measurement list at the end of the line. |
| `first_word` | 1 if the first word of the CSI data is corrupted (the chip reports this), otherwise 0. |
| number list | The actual CSI measurement: pairs of numbers (imaginary part, real part) for consecutive subcarriers. We compute the amplitude from these pairs later. |

## Transmission rules

- Over a USB port (`/dev/cu.usbmodem...`) or UART at 921600 baud or faster.
- One line per frame, ended with a newline character.
- The first line after startup is a header starting with `#`, with the column names. The program on the computer skips every line starting with `#`.
- Ordinary ESP-IDF logs would mix with the data, so it is best to turn them off. The program also ignores anything that does not start with `CSI_DATA,` just in case.

## When the program on the computer rejects a line

A line is treated as corrupted and counted in the statistics when:
- it has a different number of fields than expected,
- the count of numbers in the list does not match the `len` field,
- any number in the list is outside the range -128 to 127.

## What we do not know yet

- Exactly how many numbers the list will contain (`len`). This depends on the radio mode and will come out of the first recordings.
- Whether the order within a pair is really (imaginary, real). The documentation says so, but we will check on hardware.
- Whether the text format is too slow. If step 1.7 shows that we lose too many frames, we will switch to a binary format.
