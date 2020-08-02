FROM python:3.8-buster

COPY --from=koalaman/shellcheck:v0.7.1 /bin/shellcheck /usr/local/bin/

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]
