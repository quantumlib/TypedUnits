FROM python:3.8-buster

COPY --from=koalaman/shellcheck:v0.10.0 /bin/shellcheck /usr/local/bin/

COPY dev_tools/dev.env.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]
