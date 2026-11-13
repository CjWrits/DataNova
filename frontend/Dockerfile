# ==============================================================================
# Frontend Dockerfile - Next.js 15 SOC Analyst Console
# Node.js 20 Alpine Linux
# ==============================================================================

FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN mkdir -p /app/public

# Pass backend URL at build time for client-side API calls
ARG NEXT_PUBLIC_API_URL=https://datanova-backend-8ef2.onrender.com
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# Production runtime stage
FROM node:20-alpine AS runner

WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["sh", "-c", "npx next start -p ${PORT:-3000}"]
