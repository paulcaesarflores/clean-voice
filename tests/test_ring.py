"""The lock-free SPSC ring the device callbacks copy through."""

import threading

import numpy as np

from clean_voice.ring import Ring


def test_write_then_read_round_trips() -> None:
    ring = Ring(8)
    assert ring.write(np.arange(5, dtype=np.float32)) == 5
    out = np.zeros(5, np.float32)
    assert ring.read_into(out) == 5
    assert out.tolist() == [0, 1, 2, 3, 4]
    assert ring.available() == 0


def test_wraps_around_the_end() -> None:
    ring = Ring(8)
    out = np.zeros(6, np.float32)
    ring.write(np.arange(6, dtype=np.float32))
    ring.read_into(out)
    ring.write(np.arange(10, 16, dtype=np.float32))  # crosses the end of the buffer
    assert ring.read_into(out) == 6
    assert out.tolist() == [10, 11, 12, 13, 14, 15]


def test_never_overwrites_unread_audio() -> None:
    ring = Ring(4)
    assert ring.write(np.ones(6, np.float32)) == 4
    assert ring.space() == 0
    assert ring.write(np.ones(1, np.float32)) == 0
    out = np.zeros(6, np.float32)
    assert ring.read_into(out) == 4


def test_multichannel_frames() -> None:
    ring = Ring(5, channels=2)
    frames = np.arange(8, dtype=np.float32).reshape(4, 2)
    ring.write(frames)
    out = np.zeros((4, 2), np.float32)
    assert ring.read_into(out) == 4
    assert np.array_equal(out, frames)


def test_threads_pass_every_sample_in_order() -> None:
    ring = Ring(1000)
    total = 200_000
    received: list[np.ndarray] = []

    def produce() -> None:
        sent = 0
        while sent < total:
            chunk = np.arange(sent, min(sent + 137, total), dtype=np.float32)
            sent += ring.write(chunk)

    def consume() -> None:
        got = 0
        buf = np.zeros(211, np.float32)
        while got < total:
            n = ring.read_into(buf)
            if n:
                received.append(buf[:n].copy())
                got += n

    threads = [threading.Thread(target=produce), threading.Thread(target=consume)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    data = np.concatenate(received)
    assert np.array_equal(data, np.arange(total, dtype=np.float32))
