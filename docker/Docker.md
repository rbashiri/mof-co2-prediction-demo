# Docker setup with native MOFid

This setup builds two images and runs one application container:

- [Dockerfile.mofid](Dockerfile.mofid) builds `mofid-base:demo`, containing
  Python 3.12.3, native MOFid, and its bundled Open Babel libraries/resources.
- [Dockerfile.app](Dockerfile.app) builds the runnable web app on that base.
  It bundles a FastAPI backend, a React + Vite frontend, the five Phase 3
  Model B joblib models, and the application configuration.

MOFid source is pinned to revision
`36873683083c7bc62c1da2062df2833adc35b48a`. Package repository contents and
transitive Python dependencies are not fully locked, so this is not yet a
bit-for-bit reproducible build.

## Build

Run from the repository root with a running Docker engine:

```bash
docker build -f docker/Dockerfile.mofid -t mofid-base:demo docker
docker build -f docker/Dockerfile.app -t mof-co2-app:demo .
```

Build the MOFid image first: the application requires that local base image.
The root [.dockerignore](../.dockerignore) limits the application build
context to required files, excluding local environments and datasets.

[Dockerfile.app](Dockerfile.app) accepts `BASE_IMAGE` (default
`mofid-base:demo`) so local builds stay unchanged while CI can point at a
pushed base image.

## CI publish to Docker Hub

A GitHub Actions workflow ([`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml))
builds both images on every push and publishes them to Docker Hub.

Add these repository secrets:

- `DOCKERHUB_USERNAME`: `ripunjaykumar0902`
- `DOCKERHUB_TOKEN`: Docker Hub access token with permission to push

Both builds go to the single Hub repository `ripunjaykumar0902/mof-co2`.
The runnable app uses tags `sha-<sha>` and the branch name. Pushes to the
default branch also receive `demo` and `latest`. The MOFid base uses the
same repository with `mofid-` prefixes (`mofid-sha-<sha>`, `mofid-demo`).
You only need the app tags to pull and run.

```bash
docker pull ripunjaykumar0902/mof-co2:demo
docker run --rm -p 127.0.0.1:8501:8501 ripunjaykumar0902/mof-co2:demo
```

On this `image/publish` branch the app tag is `image-publish` until it is
merged to `main`. The first MOFid compile can take 15-30 minutes; later
runs reuse GitHub Actions cache when the base Dockerfile is unchanged.

## Native fragmentation smoke test

```bash
docker run --rm mofid-base:demo \
  /opt/mofid/bin/sbu \
  /opt/mofid/Resources/TestCIFs/P1-IRMOF-1.cif
```

Expected output includes `# Nodes:` and `# Linkers:` sections. The application
uses native fragmentation through `MOF_SBU_PATH=/opt/mofid/bin/sbu`; the full
Java-based topology workflow is not included.

## Run the web application

```bash
docker run --rm -p 127.0.0.1:8501:8501 mof-co2-app:demo
# after CI publish:
# docker run --rm -p 127.0.0.1:8501:8501 ripunjaykumar0902/mof-co2:demo
```

The FastAPI server listens on `PORT` when that environment variable is set
(Render and similar hosts), and on 8501 otherwise. Map the host port to the
same value.

Open <http://localhost:8501>. For a remote VS Code workspace, forward port
8501. Stop the foreground container with Ctrl+C; the images remain available.

Supply one CIF and its matching largest cavity diameter (angstrom), pore
limiting diameter (angstrom), void fraction (0-1), and surface area (m2/g).
The application does not calculate these four geometry values.

## Local development

```bash
# Terminal 1: API
uvicorn src.api:app --reload --port 8000

# Terminal 2: React dev server
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://localhost:8000`. For production, the container
serves the built `frontend/dist` files from FastAPI's root mount.

## Validation and remaining deployment work

Both images built successfully locally. The native IRMOF-1 smoke test
returned node and linker fragments. A terminal test inside the application
image processed hMOF-6 with LCD 10.75, PLD 9.75, void fraction 0.754903, and
surface area 3174.5, producing one finite 48-feature row and five finite
predictions. These checks establish execution for those inputs, not
prediction accuracy or generalization.

Browser upload-to-prediction, SHAP, and downloads still need verification in
the container. Hugging Face deployment is not yet validated: a Docker Space
cannot access the local `mofid-base:demo` tag. Publish an accessible base
image or provide a self-contained build, configure the Space's app port,
and adapt runtime permissions for user ID 1000 before deployment.

Review upstream license obligations before distributing the images.
