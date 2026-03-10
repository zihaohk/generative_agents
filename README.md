# Generative Agents: Interactive Simulacra of Human Behavior

<p align="center" width="100%">
  <img src="cover.png" alt="Smallville" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

This repository accompanies the paper ["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/abs/2304.03442). It contains the core simulation code, the Smallville environment, and the minimum runtime assets needed to run local simulations.

This fork is intentionally trimmed down for source control. It keeps the core code and two base simulations, while excluding large generated outputs and archival demo data.

## What This Repo Includes

- Backend simulation code in `reverie/backend_server`
- Django + Phaser frontend in `environment/frontend_server`
- Runtime map and character assets under `environment/frontend_server/static_dirs/assets`
- Two base simulations:
  - `base_the_ville_isabella_maria_klaus`
  - `base_the_ville_n25`

## What Is Not Tracked

These paths are intentionally ignored and will be generated or supplied locally:

- `reverie/backend_server/utils.py`
- `environment/frontend_server/frontend_server/settings/local.py`
- `environment/frontend_server/storage/*` except the two base simulations above
- `environment/frontend_server/compressed_storage/*`
- `environment/frontend_server/temp_storage/*`
- local SQLite files

## Prerequisites

- Python 3.9.x is the safest target for this codebase
- An OpenAI API key
- A virtual environment is strongly recommended

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Step 1: Create `utils.py`

Create `reverie/backend_server/utils.py` with the following content:

```python
openai_api_key = "<Your OpenAI API Key>"
key_owner = "<Your Name>"

maze_assets_loc = "../../environment/frontend_server/static_dirs/assets"
env_matrix = f"{maze_assets_loc}/the_ville/matrix"
env_visuals = f"{maze_assets_loc}/the_ville/visuals"

fs_storage = "../../environment/frontend_server/storage"
fs_temp_storage = "../../environment/frontend_server/temp_storage"

collision_block_id = "32125"

debug = True
```

Replace the placeholders with your own values.

## Step 2: Create `local.py`

Create `environment/frontend_server/frontend_server/settings/local.py`.

Minimal development config:

```python
DEBUG = True
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
```

This file is required in this trimmed fork because `production.py` is not included.

## Step 3: Start the Environment Server

In one terminal:

```bash
cd environment/frontend_server
python manage.py runserver
```

Then open:

- `http://localhost:8000/`
- `http://localhost:8000/simulator_home`

## Step 4: Start the Simulation Server

In a second terminal:

```bash
cd reverie/backend_server
python reverie.py
```

When prompted for the forked simulation, use one of:

```text
base_the_ville_isabella_maria_klaus
base_the_ville_n25
```

Then enter a new simulation name such as:

```text
test-simulation
```

## Step 5: Run a Simulation

At the `Enter option:` prompt, run:

```text
run 100
```

This advances the simulation by 100 steps. One in-game step is 10 seconds.

You can continue with more `run <step-count>` commands, exit without saving via `exit`, or save and exit via `fin`.

Saved simulations will appear under:

```text
environment/frontend_server/storage/<your-simulation-name>
```

## Replay and Demo

Replay and demo routes still exist, but this trimmed fork does not include the old pre-generated `July1_*` sample outputs from the original repository.

That means:

- replay works only for simulations you generated locally
- demo works only after you compress a locally generated simulation

Replay URL format:

```text
http://localhost:8000/replay/<simulation-name>/<starting-time-step>/
```

Demo URL format:

```text
http://localhost:8000/demo/<simulation-name>/<starting-time-step>/<simulation-speed>/
```

To prepare a demo, use `reverie/compress_sim_storage.py` on a simulation you already generated locally.

## Customization

### Load Initial Agent History

Two example history files are included:

- `environment/frontend_server/static_dirs/assets/the_ville/agent_history_init_n3.csv`
- `environment/frontend_server/static_dirs/assets/the_ville/agent_history_init_n25.csv`

After starting a base simulation, load history with:

```text
call -- load history the_ville/<history_file_name>.csv
```

### Create New Base Simulations

The simplest path is to copy one of the existing base simulation folders and modify it. If you change character names or exceed the map's current setup, you will likely need to edit the map in [Tiled](https://www.mapeditor.org/).

## Notes

- This codebase still uses the legacy `openai==0.27.0` Python SDK interface.
- Generated simulation data can become large quickly. This repository intentionally keeps that data out of Git.
- The included assets are enough for local runtime, not for preserving every historical artifact from the original upstream repository.

## Authors and Citation

**Authors:** Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein

Please cite the paper if you use the code or data in this repository.

```bibtex
@inproceedings{Park2023GenerativeAgents,
  author = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},
  title = {Generative Agents: Interactive Simulacra of Human Behavior},
  year = {2023},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  booktitle = {In the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)},
  keywords = {Human-AI interaction, agents, generative AI, large language models},
  location = {San Francisco, CA, USA},
  series = {UIST '23}
}
```

## Acknowledgements

Please support the artists whose work is reflected in the included assets:

- Background art: [PixyMoon (@_PixyMoon_)](https://twitter.com/_PixyMoon_)
- Furniture and interior design: [LimeZu (@lime_px)](https://twitter.com/lime_px)
- Character design: [pipohi](https://twitter.com/pipohi)
