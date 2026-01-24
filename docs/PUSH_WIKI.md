# Pushing these wiki pages to GitHub Wiki

GitHub Wiki is stored in a separate repository named `<your-repo>.wiki.git`.

Example:
```bash
git clone https://github.com/<ORG_OR_USER>/<REPO>.wiki.git
```

Then copy the pages from this repository:
```bash
cp -r docs/wiki/* <REPO>.wiki/
```

Commit and push:
```bash
cd <REPO>.wiki
git add .
git commit -m "Update wiki"
git push
```

Tip: if you prefer not to use GitHub Wiki, you can keep these pages as versioned documentation in `docs/wiki/` and link to them from the README.
