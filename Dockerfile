ARG FRESCOBENCH_DEPS_IMAGE=python:3.11-slim
FROM ${FRESCOBENCH_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/frescobench
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 FRESCOBENCH_DATA=/var/lib/frescobench
WORKDIR /opt/frescobench
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY frescobench ./frescobench
COPY static ./static
RUN useradd -U -u 11519 frescobench && install -d -o frescobench -g frescobench /var/lib/frescobench
USER frescobench
EXPOSE 8000
CMD ["python", "-m", "frescobench"]
