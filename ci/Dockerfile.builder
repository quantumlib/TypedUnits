FROM python:3.7-buster

# can't just give "--one-top-level=shellcheck" to tar because shellcheck
# folder has a file named shellcheck, so tar complains about the file
# existing when trying to make the directory
RUN mkdir shellcheck \
  && wget -qO- "https://storage.googleapis.com/qh-build-cloud-build-packages/shellcheck-v0.7.0.linux.x86_64.tar.xz" \
  | tar -xJ -C shellcheck --strip-components 1 \
  && mv shellcheck/shellcheck /usr/local/bin/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]
