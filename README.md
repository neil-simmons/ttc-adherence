---
title: TTC-ScheduleWatch
emoji: 🚊
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# TTC-ScheduleWatch

## Running Locally

The live app is available at **https://huggingface.co/spaces/neil-simmons/TTC-ScheduleWatch** and requires no setup.
If you need to run the project locally, choose one of the two options below.

---

### Option A — Docker (Recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) with the **containerd image store** enabled
(`Docker Desktop → Settings → General → Use containerd for pulling and storing images`).

```bash
docker run -it -p 7860:7860 --platform=linux/amd64 \
  registry.hf.space/neil-simmons-ttc-schedulewatch:latest \
  streamlit run app.py
```

Then open **http://localhost:7860** in your browser.

This runs the exact container deployed on HuggingFace Spaces — no additional configuration required.

---

### Option B — Manual Installation

**Requires Python 3.11.** Other versions are not supported.

**1. Verify your Python version**

```bash
python3.11 --version
```

If `python3.11` is not found, download Python 3.11 from [python.org/downloads](https://www.python.org/downloads/)
and ensure it is on your PATH before continuing.

**2. Clone the repository**

```bash
git clone https://github.com/neil-simmons/TTC-ScheduleWatch.git
cd TTC-ScheduleWatch
```

**3. Create a virtual environment using Python 3.11 explicitly**

```bash
python3.11 -m venv venv
```

> On Windows, use `py -3.11 -m venv venv` instead.

**4. Activate the virtual environment**

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the app**

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

> **Note:** On first launch, the app downloads GTFS and AVL data files
> from HuggingFace (~few minutes depending on connection speed).
> For Option A (Docker), this download happens every time the container starts.
> For Option B (manual), files are cached locally after the first download.




© 2026 Neil Simmons. All Rights Reserved.
This source code is provided publicly solely for the purpose of evaluation for the Transit Data Challenge 2026. No license is granted to any person or entity to copy, modify, distribute, or use this code for any other purpose.
Data Attribution: The data used by this application contains information licensed under the Open Government Licence – Toronto.
