FROM ydigital.azurecr.io/y-base-builder:v3.0 AS builder

WORKDIR /install
ADD requirements-git.txt .
RUN pip3 install wheel
RUN mkdir -p $HOME/.ssh
RUN ssh-keyscan -t rsa ssh.dev.azure.com > $HOME/.ssh/known_hosts
RUN --mount=type=ssh pip3 wheel -r requirements-git.txt --wheel-dir=/install/wheels

FROM ydigital.azurecr.io/y-base-run:v2.0
RUN useradd yvette --uid 10000
COPY --from=builder /install/wheels /install/wheels
ADD requirements.txt .
RUN pip3 install --find-links=/install/wheels -r requirements.txt
WORKDIR /project
ADD ./src /project
RUN chown -R yvette:yvette /project
USER yvette
# CMD ["python3", "main.py", "$MODE"]
CMD python3 main.py $MODE