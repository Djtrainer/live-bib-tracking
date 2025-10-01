# Multi-stage build for optimized production image
# Stage 1: Build stage
FROM --platform=linux/amd64 node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./
COPY .nvmrc ./

# Install dependencies with npm ci for faster, reliable builds
RUN npm ci --silent && \
    npm cache clean --force

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Stage 2: Production stage
FROM --platform=linux/amd64 node:18-alpine AS production

# Install serve to serve static files
RUN npm install -g serve@14.2.3 && \
    npm cache clean --force

# Set working directory
WORKDIR /app

# Copy built application from builder stage
COPY --from=builder /app/dist ./dist

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Change ownership of the app directory
RUN chown -R nextjs:nodejs /app
USER nextjs

# Set port environment variable
ENV PORT=5173

# Expose port
EXPOSE 5173

# Start production server
CMD ["serve", "-s", "dist", "-l", "5173"]

# Stage 3: Development stage (default)
FROM --platform=linux/amd64 node:18-alpine AS development

# Set working directory
WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./
COPY .nvmrc ./

# Install dependencies with npm ci for faster, reliable builds
RUN npm ci --silent && \
    npm cache clean --force

# Copy source code
COPY . .

# Set port environment variable
ENV PORT=5173

# Expose port 5173 (Vite default)
EXPOSE 5173

# Start development server
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
