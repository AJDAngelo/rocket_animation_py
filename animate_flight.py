"""Turn an OpenRocket simulation CSV export into a launch video (MP4).

Usage:
    ./venv/bin/python animate_flight.py flight.csv            -> flight.mp4
    ./venv/bin/python animate_flight.py flight.csv --speed 2  -> plays 2x faster
    ./venv/bin/python animate_flight.py --demo                -> built-in sample flight

Export from OpenRocket: Flight simulations -> select sim -> Plot / export ->
Export data -> CSV (comma separated, keep "include flight events" checked).
"""
import argparse
import re
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon, Wedge

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

FPS = 30


# ---------------------------------------------------------------- data loading

def find_col(header, *keys):
    for key in keys:
        for i, name in enumerate(header):
            if key.lower() in name.lower():
                return i
    return None


def load_openrocket_csv(path):
    header, rows, events = None, [], {}
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if re.search(r"written for 0 variables", text):
        sys.exit("This CSV has no data columns ('written for 0 variables').\n"
                 "Re-export it from OpenRocket with the variables checked: in Export data,\n"
                 "click 'Select all' in the 'Variables to export' list (at minimum Time,\n"
                 "Altitude, Vertical velocity, Total velocity, Lateral distance, Thrust).")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            m = re.search(r"Event\s+(\w+)\s+occurred at t=([\d.]+)", line)
            if m:
                events.setdefault(m.group(1).upper(), float(m.group(2)))
            elif "Time" in line and header is None:
                header = [h.strip() for h in line.lstrip("#").split(",")]
            continue
        parts = line.split(",")
        try:
            rows.append([float(p) if p.strip() not in ("", "NaN") else np.nan for p in parts])
        except ValueError:
            continue
    if header is None or not rows:
        sys.exit("Could not find data in the CSV. Make sure it was exported as comma-separated CSV.")

    data = np.array(rows, dtype=float)

    def col(*keys, default=None):
        i = find_col(header, *keys)
        if i is None:
            return default
        return np.nan_to_num(data[:, i])

    t = col("Time")
    alt = col("Altitude")
    if t is None or alt is None:
        sys.exit("The CSV needs at least the 'Time' and 'Altitude' columns.")
    x = col("Lateral distance", "Position East", default=np.zeros_like(t))
    vz = col("Vertical velocity", default=np.gradient(alt, t))
    v = col("Total velocity", default=np.abs(vz))
    thrust = col("Thrust", default=None)
    return dict(t=t, alt=alt, x=x, v=v, vz=vz, thrust=thrust, events=events)


def demo_flight():
    """Simple 2D physics model of Test_Rocket.ork on an Estes C6-5."""
    curve_t = [0, 0.2, 0.3, 0.4, 1.85, 1.86]
    curve_f = [0, 14.0, 6.0, 4.6, 4.6, 0]
    dry, prop, burn = 0.048, 0.0124, 1.85
    cd_a = 0.45 * np.pi * 0.0124 ** 2
    chute_cd_a = 0.8 * np.pi * 0.15 ** 2
    dt, g, rho = 0.01, 9.81, 1.225
    t, x, z, vx, vz = 0.0, 0.0, 0.0, 0.0, 0.0
    out = {k: [] for k in ("t", "alt", "x", "v", "vz", "thrust")}
    events = {"LAUNCH": 0.0}
    chute = False
    while True:
        f = np.interp(t, curve_t, curve_f)
        m = dry + prop * max(0.0, 1 - t / burn)
        v = np.hypot(vx, vz)
        if chute:
            # parachute: drag relative to air moving with 2 m/s wind
            rvx, rvz = vx - 2.0, vz
            rv = np.hypot(rvx, rvz) or 1e-9
            d = 0.5 * rho * rv ** 2 * (cd_a + chute_cd_a)
            ax, az = -d * rvx / rv / m, -d * rvz / rv / m - g
        else:
            ux, uz = (vx / v, vz / v) if v > 1 else (np.sin(np.radians(2)), np.cos(np.radians(2)))
            d = 0.5 * rho * v ** 2 * cd_a
            ax = (f - d) * ux / m
            az = (f - d) * uz / m - g if (z > 0 or f > m * g) else 0.0
        vx, vz = vx + ax * dt, vz + az * dt
        x, z, t = x + vx * dt, z + vz * dt, t + dt
        if "BURNOUT" not in events and t >= burn:
            events["BURNOUT"] = t
        if "APOGEE" not in events and t > 0.5 and vz < 0:
            events["APOGEE"] = t
        if not chute and t >= burn + 5.0:
            chute = True
            events["EJECTION_CHARGE"] = events["RECOVERY_DEVICE_DEPLOYMENT"] = t
        for k, val in zip(out, (t, max(z, 0), x, np.hypot(vx, vz), vz, f)):
            out[k].append(val)
        if t > 1 and z <= 0:
            events["GROUND_HIT"] = t
            break
    d = {k: np.array(val) for k, val in out.items()}
    d["events"] = events
    return d


# ---------------------------------------------------------------- drawing

def rocket_shape(cx, cy, angle, size):
    """Rocket outline pointing along `angle` (radians from +y), centred at cx, cy."""
    body = np.array([[0, 1.0], [0.12, 0.65], [0.12, -0.6], [0.32, -1.0], [0.12, -0.85],
                     [-0.12, -0.85], [-0.32, -1.0], [-0.12, -0.6], [-0.12, 0.65]]) * size
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, s], [-s, c]])
    return body @ rot.T + [cx, cy]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="OpenRocket CSV export")
    ap.add_argument("--demo", action="store_true", help="use a built-in sample flight")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (2 = twice as fast)")
    ap.add_argument("-o", "--output", help="output .mp4 path")
    ap.add_argument("--title", default="Test Rocket - Estes C6-5")
    args = ap.parse_args()

    if args.demo:
        d, out = demo_flight(), Path(args.output or "demo_flight.mp4")
    elif args.csv:
        d, out = load_openrocket_csv(args.csv), Path(args.output or Path(args.csv).with_suffix(".mp4"))
    else:
        ap.error("give a CSV file or --demo")

    t, alt, x, v, vz, ev = d["t"], d["alt"], d["x"], d["v"], d["vz"], d["events"]
    burnout = ev.get("BURNOUT", t[np.argmax(d["thrust"] < 0.01 * d["thrust"].max())] if d["thrust"] is not None else 1.5)
    deploy = ev.get("RECOVERY_DEVICE_DEPLOYMENT", ev.get("EJECTION_CHARGE", np.inf))
    apogee_i = int(np.argmax(alt))
    t_end = t[-1]

    # ---- figure layout
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0b1020")
    fig.suptitle(args.title, color="white", fontsize=18, fontweight="bold", y=0.97)
    ax = fig.add_axes([0.07, 0.08, 0.50, 0.82])
    ax2 = fig.add_axes([0.63, 0.50, 0.34, 0.36], facecolor="#111a33")
    ax3 = fig.add_axes([0.63, 0.08, 0.34, 0.30], facecolor="#0b1020")
    ax3.axis("off")

    top = alt.max() * 1.15 + 5
    xspan = max(np.ptp(x) * 1.3, top * 0.6)
    xmid = (x.max() + x.min()) / 2
    ax.set_xlim(xmid - xspan / 2, xmid + xspan / 2)
    ax.set_ylim(-top * 0.04, top)
    ax.set_aspect("auto")
    sky = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(sky, extent=[*ax.get_xlim(), 0, top], origin="lower", cmap="Blues_r",
              aspect="auto", alpha=0.85, zorder=0)
    ax.axhspan(-top, 0, color="#2e6b2e", zorder=1)
    ax.set_ylabel("Altitude (m)", color="white")
    ax.set_xlabel("Downrange distance (m)", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#445")
    ax.plot([x[0], x[0]], [0, top * 0.04], color="#bbb", lw=2, zorder=2)  # launch rod
    ax.axhline(alt.max(), color="#1e2a44", ls=":", lw=0.8, alpha=0.6, zorder=2)
    ax.text(ax.get_xlim()[0] + xspan * 0.02, alt.max(), f" apogee {alt.max():.0f} m",
            color="#1e2a44", va="bottom", fontsize=10, zorder=2)

    trail, = ax.plot([], [], color="#555", lw=1.2, alpha=0.6, zorder=3)
    size = top * 0.035
    rocket = Polygon(rocket_shape(0, 0, 0, size), closed=True, fc="#d63031", ec="#2d3436", lw=1.5, zorder=5)
    flame = Polygon([[0, 0]] * 3, closed=True, fc="#ffb000", ec="#ff5500", zorder=4)
    chute = Wedge((0, 0), size * 1.4, 0, 180, fc="#ff4f6d", ec="white", lw=1, zorder=6)
    lines = [ax.plot([], [], color="#2d3436", lw=0.8, zorder=5)[0] for _ in range(2)]
    for a in (rocket, flame, chute):
        ax.add_patch(a)

    ax2.set_xlim(0, t_end)
    ax2.set_ylim(0, top)
    ax2.set_title("Altitude vs time", color="white", fontsize=11)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.set_xlabel("Time (s)", color="white", fontsize=9)
    for sp in ax2.spines.values():
        sp.set_color("#445")
    ax2.plot(t, alt, color="#556", lw=1)
    alt_line, = ax2.plot([], [], color="#4fc3f7", lw=2)
    alt_dot, = ax2.plot([], [], "o", color="#4fc3f7")

    readout = ax3.text(0, 1, "", color="white", fontsize=15, family="monospace", va="top")
    phase_txt = ax3.text(0, 0.05, "", color="#ffcc33", fontsize=20, fontweight="bold")

    # ---- frames: real time (x --speed) through deployment, then the long
    # parachute descent is fast-forwarded so it fills about 12 s of video
    t_fast = min(deploy + 2.0, t_end)
    slow = np.arange(0, t_fast, args.speed / FPS)
    descent_speed = max(args.speed, (t_end - t_fast) / 12.0)
    fast = np.arange(t_fast, t_end, descent_speed / FPS)
    frame_t = np.concatenate([slow, fast, np.full(FPS * 2, t_end)])  # 2 s hold at the end
    n_frames = len(frame_t)
    ff_txt = ax3.text(0, -0.12, "", color="#aaa", fontsize=11)

    def phase(tt):
        if tt >= t_end - 1e-6:
            return "LANDED"
        if tt < burnout:
            return "POWERED ASCENT"
        if tt >= deploy:
            return "PARACHUTE DESCENT"
        if tt < t[apogee_i]:
            return "COASTING"
        return "APOGEE" if tt - t[apogee_i] < 1.0 else "FALLING"

    def draw(tt):
        i = int(np.searchsorted(t, tt).clip(0, len(t) - 1))
        cx, cy = x[i], alt[i]
        ph = phase(tt)
        under_chute = tt >= deploy
        if under_chute:
            ang, ry = np.pi, cy  # hangs nose-down under the chute
            chute.set_center((cx, cy + size * 2.4))
            chute.set_visible(True)
            for k, ln in enumerate(lines):
                ln.set_data([cx + (k * 2 - 1) * size * 1.3, cx], [cy + size * 2.4, cy + size * 0.9])
                ln.set_visible(True)
        else:
            j = min(i + 1, len(t) - 1)
            dx, dz = x[j] - x[max(i - 1, 0)], alt[j] - alt[max(i - 1, 0)]
            ang = np.arctan2(dx, dz) if np.hypot(dx, dz) > 1e-6 else 0.0
            ry = cy + size  # base of the rocket sits at the data point
            chute.set_visible(False)
            for ln in lines:
                ln.set_visible(False)
        rocket.set_xy(rocket_shape(cx, ry, ang, size))
        if tt < burnout:
            flick = 1.0 + 0.3 * np.sin(tt * 90)
            base = rocket_shape(cx, ry, ang, size)
            tail = np.array([-np.sin(ang), -np.cos(ang)]) * size * 0.9 * flick
            nozzle = (base[4] + base[5]) / 2
            flame.set_xy([base[4], base[5], nozzle + tail])
            flame.set_visible(True)
        else:
            flame.set_visible(False)
        trail.set_data(x[: i + 1], alt[: i + 1])
        alt_line.set_data(t[: i + 1], alt[: i + 1])
        alt_dot.set_data([t[i]], [alt[i]])
        readout.set_text(f"Time      {tt:6.1f} s\n"
                         f"Altitude  {alt[i]:6.1f} m\n"
                         f"Speed     {v[i]:6.1f} m/s\n"
                         f"Vert. vel {vz[i]:+6.1f} m/s\n"
                         f"Max alt   {alt[: i + 1].max():6.1f} m")
        phase_txt.set_text(ph)
        ff = t_fast <= tt < t_end and descent_speed > args.speed * 1.5
        ff_txt.set_text(f">> fast-forward {descent_speed:.0f}x" if ff else "")

    writer = FFMpegWriter(fps=FPS, bitrate=4000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    print(f"Rendering {n_frames} frames ({n_frames / FPS:.0f} s of video)...")
    with writer.saving(fig, str(out), dpi=100):
        for k, tt in enumerate(frame_t):
            draw(tt)
            writer.grab_frame(facecolor=fig.get_facecolor())
            if k % (FPS * 5) == 0:
                print(f"  {k}/{n_frames}")
    print(f"Saved {out.resolve()}")
    print(f"Apogee {alt.max():.1f} m at t={t[apogee_i]:.1f} s, landed at t={t_end:.1f} s")


if __name__ == "__main__":
    main()
