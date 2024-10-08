FROM python:3.10-buster

COPY --from=koalaman/shellcheck:v0.10.0 /bin/shellcheck /usr/local/bin/

COPY dev_tools/protos.txt .
COPY dev_tools/dev.env.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]
