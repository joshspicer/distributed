FROM mcr.microsoft.com/devcontainers/typescript-node:16
COPY /homomorphic-enc /project
WORKDIR /project
RUN yarn
RUN yarn build
ENTRYPOINT [ "/usr/local/bin/node", "/project/dist/main.js" ]

