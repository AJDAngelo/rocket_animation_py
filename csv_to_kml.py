import sys
from datetime import datetime, timedelta, timezone

src = sys.argv[1]
out = src.rsplit(".", 1)[0] + ".kml"

# Read the OpenRocket CSV: the column names are on a line starting with "#"
header, rows = None, []
for line in open(src, encoding="utf-8", errors="replace"):
    line = line.strip()
    if line.startswith("#"):
        if "Time" in line and header is None:
            header = [h.strip() for h in line.lstrip("#").split(",")]
    elif line:
        rows.append(line.split(","))
if header is None:
    sys.exit("No column names found - export again with variables checked.")

def col(name):
    for i, h in enumerate(header):
        if h.startswith(name):
            return i
    sys.exit(f"Column '{name}' not found - check it when exporting.")

it, ialt, ilat, ilon = col("Time"), col("Altitude"), col("Latitude"), col("Longitude")

# Turn each row into a timestamp and a "longitude latitude altitude" point
start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
whens, coords = [], []
for r in rows:
    try:
        t, alt, lat, lon = (float(r[i]) for i in (it, ialt, ilat, ilon))
    except (ValueError, IndexError):
        continue
    whens.append((start + timedelta(seconds=t)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z")
    coords.append(f"{lon:.7f} {lat:.7f} {alt:.1f}")

# Write two things: the full flight path as a red line,
# and a moving "Rocket" marker that follows it in time
kml = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
<Document>
  <name>Rocket flight</name>
  <Style id="path">
    <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>
    <PolyStyle><color>400000ff</color></PolyStyle>
  </Style>
  <Placemark><name>Flight path</name><styleUrl>#path</styleUrl>
    <LineString><extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode>
      <coordinates>{" ".join(c.replace(" ", ",") for c in coords)}</coordinates>
    </LineString>
  </Placemark>
  <Placemark><name>Rocket</name>
    <gx:Track><altitudeMode>relativeToGround</altitudeMode>
      {"".join(f"<when>{w}</when>" for w in whens)}
      {"".join(f"<gx:coord>{c}</gx:coord>" for c in coords)}
    </gx:Track>
  </Placemark>
</Document>
</kml>'''

open(out, "w").write(kml)
print(f"Saved {out} ({len(coords)} points)")

