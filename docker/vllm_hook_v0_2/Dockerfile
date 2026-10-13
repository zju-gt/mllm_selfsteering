FROM vllm/vllm-openai:v0.12.0

USER root

ARG VLLM_HOOK_REVISION=0e34fddf2c234a50c9c820c2b7852472f8f4c51c

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/IBM/vLLM-Hook.git /opt/vLLM-Hook \
    && cd /opt/vLLM-Hook \
    && git checkout "${VLLM_HOOK_REVISION}" \
    && python3 -m pip install --no-cache-dir zstandard \
    && python3 -m pip install --no-cache-dir --no-deps -e ./vllm_hook_plugins \
    && python3 -c "import importlib.metadata as m, vllm, vllm_hook_plugins; assert vllm.__version__ == '0.12.0'; assert m.version('vllm-hook-plugins') == '0.2.0'"

# v0.2 uses the legacy vLLM V1 runner and this environment variable. Its
# plugin forces eager execution when it injects the steering worker.
ENV VLLM_USE_V1=1 \
    VLLM_USE_AOT_COMPILE=0 \
    VLLM_HOOK_WORKER=steer

WORKDIR /workspace/mllm_selfsteering

EXPOSE 8770

ENTRYPOINT []
CMD ["bash"]
