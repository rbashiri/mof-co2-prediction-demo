# Docker setup with native MOFid

This setup builds two images and runs one application container:

- [Dockerfile.mofid](Dockerfile.mofid) builds `mofid-base:demo`, containing
  Python 3.12.3, native MOFid, and its bundled Open Babel libraries/resources.
- [Dockerfile.app](Dockerfile.app) builds `mof-co2-app:demo` on that base,
  adding Zeo++ 0.3, the Streamlit application, configuration, and five models.

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
`mofid-base:demo`) so local builds stay unchanged while CI can point at the
pushed base image.

## CI publish to Docker Hub

A GitHub Actions workflow ([`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml))
builds both images on every push and publishes them to Docker Hub
(`ripunjaykumar0902/mof-co2`) using repository secrets `DOCKERHUB_USERNAME`
and `DOCKERHUB_TOKEN`. For branch `osi/app`, the published app image tag is
`ripunjaykumar0902/mof-co2:osi-app` (plus `sha-<commit>`).

The Streamlit server listens on `PORT` when set (Render and similar hosts) and
on `8501` otherwise.

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
```

Open <http://localhost:8501>. For a remote VS Code workspace, forward port
8501. Stop the foreground container with Ctrl+C; the images remain available.

Supply one CIF. Zeo++ automatically calculates largest cavity diameter, pore
limiting diameter, total probe-occupiable void fraction, and total surface
area. MOFid/RDKit and pymatgen derive the remaining descriptors. The image
sets `ZEO_NETWORK_PATH=/opt/zeopp/network`; native MOFid is supplied by the
base image.

To enable the separately requested Gemini explanation, pass credentials only
at runtime:

```bash
docker run --rm -p 127.0.0.1:8501:8501 \
  -e GEMINI_ENABLED=true \
  -e GEMINI_API_KEY \
  -e GEMINI_MODEL=gemini-3.5-flash \
  mof-co2-app:demo
```

The app remains fully usable without these variables.

## Validation and remaining deployment work

The native IRMOF-1 smoke test previously returned node and linker fragments.
The Zeo++ hackathon image also processed committed sample CIFs through
automatic geometry, 48-feature construction, five predictions, SHAP, and CSV
checks. Re-run those checks for this Streamlit image after building it:

```bash
docker run --rm -p 127.0.0.1:8501:8501 mof-co2-app:demo
```

Open <http://localhost:8501>, upload
`notebook/sample/cif_files/hMOF-0.cif`, and verify four geometry values, five
finite predictions, SHAP, and CSV download. Confirm invalid CIFs fail without
substituting descriptors. These checks establish execution, not prediction
accuracy or generalization.

Hugging Face deployment is not yet validated: a Docker Space
cannot access the local `mofid-base:demo` tag. Publish an accessible base
image or provide a self-contained build, configure the Space's app port,
and adapt runtime permissions for user ID 1000 before deployment.

Review upstream license obligations before distributing the images.