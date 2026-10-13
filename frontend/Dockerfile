# Frontend image: build the bundle, then serve it with nginx.
#
# Stage 1 builds with node_modules present; stage 2 ships only the static
# output, so neither node nor the dev dependencies reach the runtime image.

# --- build -----------------------------------------------------------------
FROM node:22-slim AS build

WORKDIR /app

# Copy manifests first so `npm ci` is cached until the lockfile changes.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# VITE_API_URL is baked into the bundle at build time. It defaults to the
# same-origin /api path so nginx can proxy it and no rebuild is needed when the
# backend URL changes.
ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}

COPY frontend/ ./

RUN npm run build

# --- serve -----------------------------------------------------------------
FROM nginx:1.27-alpine AS serve

# Replace the stock config with one that serves the SPA and proxies the API.
RUN rm -f /etc/nginx/conf.d/default.conf
COPY frontend/nginx.conf /etc/nginx/conf.d/lumina.conf

COPY --from=build /app/dist /usr/share/nginx/html

# nginx keeps its default root here on purpose. Dropping to the unprivileged
# nginx user also requires relocating the pid file and every temp path, which
# needs a replacement main nginx.conf; the stock setup is the tested one.
# Listening on 8080 rather than 80 means no capability is needed for the port.
# The api container, which is the one that runs model code, is non-root.

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/ >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]