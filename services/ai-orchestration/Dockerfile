# AI Orchestration Service Dockerfile
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY services/ai-orchestration/package*.json ./
COPY services/ai-orchestration/tsconfig.json ./

# Copy shared dependency from context root
COPY shared ./shared

# Install all dependencies (including dev for building)
RUN npm ci

# Copy source code
COPY services/ai-orchestration/src/ ./src/

# Build the application
RUN npm run build

# Production stage
FROM node:20-alpine AS production

# Create app user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S userwhisperer -u 1001

# Set working directory
WORKDIR /app

# Copy package files
COPY services/ai-orchestration/package*.json ./

# Install only production dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy built application from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/shared ./shared

# Change ownership to app user
RUN chown -R userwhisperer:nodejs /app
USER userwhisperer

# Expose port
EXPOSE 3006

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3006/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"

# Start the application
CMD ["node", "dist/index.js"]
