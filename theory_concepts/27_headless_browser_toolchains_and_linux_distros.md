# Concept 27: Headless Browser Toolchains & Linux Distro Compatibility

## 1. Upstream Package Manager Assumptions
- Many Python browser automation CLIs (such as `playwright install-deps` and `scrapling install`) assume Debian/Ubuntu environments with `apt-get`.
- On Red Hat/Fedora/CentOS distributions (`dnf`) or Arch Linux (`pacman`), running `install-deps` triggers `apt-get: command not found` with `subprocess.CalledProcessError`.

## 2. User-Space Binary Isolation
- Browser automation runtimes (Playwright, Camoufox) do not require system root installation.
- Executing:
  ```bash
  playwright install firefox chromium
  python -m camoufox fetch
  ```
  downloads pre-compiled standalone browser binaries directly into user-space caches (`~/.cache/ms-playwright` and `~/.cache/camoufox`), eliminating OS package manager collisions.

## 3. Graceful Fallback Strategy
Always wrap headless browser invocations (`StealthyFetcher`) in a try-except fallback to standard HTTP fetchers (`Fetcher`). If a graphics library is missing or a display fails to allocate, the agent pipeline continues uninterrupted using direct HTTP.
