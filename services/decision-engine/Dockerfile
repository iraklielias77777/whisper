# Decision Engine Service Dockerfile
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY services/decision-engine/package*.json ./
COPY services/decision-engine/tsconfig.json ./

# Copy shared dependency from context root
COPY shared ./shared

# Install all dependencies (including dev for building)
RUN npm ci

# Copy source code
COPY services/decision-engine/src/ ./src/

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
COPY services/decision-engine/package*.json ./

# Install only production dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy built application from builder stage
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/shared ./shared

# Change ownership to app user
RUN chown -R userwhisperer:nodejs /app
USER userwhisperer

# Expose port
EXPOSE 3003

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3003/health', (res) => { process.exit(res.statusCode === 200 ? 0 : 1) })"

# Start the application
CMD ["node", "dist/index.js"]
